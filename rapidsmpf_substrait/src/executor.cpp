/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/executor.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>

#include <rmm/mr/per_device_resource.hpp>

// Undefine DEBUG to avoid conflict with any -DDEBUG flag
#ifdef DEBUG
#undef DEBUG
#endif

#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/table/table_view.hpp>

namespace rapidsmpf_substrait {

// Helper to get integer from environment variable with default
static int get_env_int(const char* name, int default_value) {
    const char* val = std::getenv(name);
    if (val) {
        try {
            return std::stoi(val);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

Executor::Executor() {
    // Get the current device memory resource
    auto* current_mr = rmm::mr::get_current_device_resource();

    // Read configuration from environment variables
    int num_streaming_threads = get_env_int("NUM_STREAMING_THREADS", 4);
    int num_streams = get_env_int("NUM_STREAMS", 16);
    
    // Print configuration
    std::cout << "Executor configuration:" << std::endl;
    std::cout << "  NUM_STREAMING_THREADS: " << num_streaming_threads << std::endl;
    std::cout << "  NUM_STREAMS: " << num_streams << std::endl;
    
    // Check for other relevant env vars
    const char* kvikio = std::getenv("KVIKIO_NTHREADS");
    const char* cudf_workers = std::getenv("LIBCUDF_NUM_HOST_WORKERS");
    if (kvikio) std::cout << "  KVIKIO_NTHREADS: " << kvikio << std::endl;
    if (cudf_workers) std::cout << "  LIBCUDF_NUM_HOST_WORKERS: " << cudf_workers << std::endl;

    // Create environment configuration
    std::unordered_map<std::string, std::string> environment;
    environment["NUM_STREAMING_THREADS"] = std::to_string(num_streaming_threads);
    auto options = rapidsmpf::config::Options(environment);

    // Create communicator for single-process execution
    comm_ = std::make_shared<rapidsmpf::Single>(options);

    // Create RmmResourceAdaptor as member to ensure it lives long enough
    mr_adaptor_ = std::make_unique<rapidsmpf::RmmResourceAdaptor>(current_mr);

    // Create statistics
    statistics_ = std::make_shared<rapidsmpf::Statistics>(mr_adaptor_.get());

    // Create stream pool with configurable size
    stream_pool_ = std::make_shared<rmm::cuda_stream_pool>(num_streams);

    // Create buffer resource
    br_ = std::make_shared<rapidsmpf::BufferResource>(
        mr_adaptor_.get(),
        rapidsmpf::BufferResource::PinnedMemoryResourceDisabled,
        std::unordered_map<rapidsmpf::MemoryType, rapidsmpf::BufferResource::MemoryAvailable>{},
        std::nullopt,
        stream_pool_,
        statistics_
    );

    // Create streaming context
    ctx_ = std::make_shared<rapidsmpf::streaming::Context>(options, comm_, br_, statistics_);
}

Executor::Executor(
    std::shared_ptr<rapidsmpf::Communicator> comm,
    std::shared_ptr<rapidsmpf::BufferResource> br
) : comm_(std::move(comm)), br_(std::move(br)) {
    int num_streaming_threads = get_env_int("NUM_STREAMING_THREADS", 4);
    
    std::unordered_map<std::string, std::string> environment;
    environment["NUM_STREAMING_THREADS"] = std::to_string(num_streaming_threads);
    auto options = rapidsmpf::config::Options(environment);

    auto statistics = rapidsmpf::Statistics::disabled();
    ctx_ = std::make_shared<rapidsmpf::streaming::Context>(options, comm_, br_, statistics);
}

Executor::~Executor() = default;

// Sink coroutine that collects result tables
static rapidsmpf::streaming::Node collect_results(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::vector<std::unique_ptr<cudf::table>>& results,
    std::atomic<size_t>& row_count
) {
    rapidsmpf::streaming::ShutdownAtExit closer{ch_in};

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();

        auto chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
            msg.release<rapidsmpf::streaming::TableChunk>()
        );

        // Ensure data is on device
        if (!chunk->is_available()) {
            auto [reservation, overbooking] = ctx->br()->reserve(
                rapidsmpf::MemoryType::DEVICE,
                chunk->make_available_cost(),
                true
            );
            *chunk = chunk->make_available(reservation);
        }

        row_count += chunk->table_view().num_rows();

        // Copy the table for the result
        auto stream = chunk->stream();
        auto mr = ctx->br()->device_mr();
        auto table = std::make_unique<cudf::table>(chunk->table_view(), stream, mr);
        results.push_back(std::move(table));
    }
}

ExecutionResult Executor::Execute(std::unique_ptr<PhysicalOperator> plan) {
    ExecutionResult result;

    if (!plan) {
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Build the streaming pipeline from the operator tree
    std::vector<rapidsmpf::streaming::Node> nodes;

    // Collect operators in execution order (post-order traversal)
    std::vector<PhysicalOperator*> operators;
    std::function<void(PhysicalOperator*)> collect = [&](PhysicalOperator* op) {
        for (auto& child : op->Children()) {
            collect(child.get());
        }
        operators.push_back(op);
    };
    collect(plan.get());

    if (operators.empty()) {
        return result;
    }

    // For a simple linear plan, create channels between consecutive operators
    std::vector<std::shared_ptr<rapidsmpf::streaming::Channel>> channels;
    for (size_t i = 0; i <= operators.size(); i++) {
        channels.push_back(ctx_->create_channel());
    }

    // Build nodes for each operator
    for (size_t i = 0; i < operators.size(); i++) {
        auto* op = operators[i];

        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in = nullptr;
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out = channels[i];

        if (!op->IsSource() && i > 0) {
            ch_in = channels[i - 1];
        }

        nodes.push_back(op->BuildNode(ctx_, ch_in, ch_out));
    }

    // Add a sink node to collect results
    std::vector<std::unique_ptr<cudf::table>> result_tables;
    std::atomic<size_t> row_count{0};
    nodes.push_back(collect_results(ctx_, channels[operators.size() - 1], result_tables, row_count));

    // Execute the pipeline
    rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));

    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.rows_processed = row_count.load();

    // Concatenate all result tables into one
    if (!result_tables.empty()) {
        std::vector<cudf::table_view> views;
        views.reserve(result_tables.size());
        for (auto const& t : result_tables) {
            views.push_back(t->view());
        }
        result.table = cudf::concatenate(views);
    }

    return result;
}

std::unique_ptr<cudf::table> Executor::ExecuteAndGetTable(
    std::unique_ptr<PhysicalOperator> plan
) {
    auto result = Execute(std::move(plan));
    return std::move(result.table);
}

}  // namespace rapidsmpf_substrait
