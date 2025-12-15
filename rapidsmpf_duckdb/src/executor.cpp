/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/executor.hpp"

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/per_device_resource.hpp>

// Undefine DEBUG to avoid conflict with DuckDB's -DDEBUG flag and rapidsmpf's LOG_LEVEL::DEBUG
#ifdef DEBUG
#undef DEBUG
#endif

#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include <cudf/interop.hpp>

#include <atomic>
#include <iostream>

namespace rapidsmpf_duckdb {

Executor::Executor() {
    // Get the current device memory resource
    auto* current_mr = rmm::mr::get_current_device_resource();
    
    // Create environment configuration
    std::unordered_map<std::string, std::string> environment;
    environment["NUM_STREAMING_THREADS"] = "4";  // Single thread for now
    auto options = rapidsmpf::config::Options(environment);
    
    // Create communicator for single-process execution
    comm_ = std::make_shared<rapidsmpf::Single>(options);
    
    // Create RmmResourceAdaptor as member to ensure it lives long enough
    mr_adaptor_ = std::make_unique<rapidsmpf::RmmResourceAdaptor>(current_mr);
    
    // Create statistics
    statistics_ = std::make_shared<rapidsmpf::Statistics>(mr_adaptor_.get());
    
    // Create stream pool as member
    stream_pool_ = std::make_shared<rmm::cuda_stream_pool>(4, rmm::cuda_stream::flags::non_blocking);
    
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
    std::unordered_map<std::string, std::string> environment;
    environment["NUM_STREAMING_THREADS"] = "4";
    auto options = rapidsmpf::config::Options(environment);
    
    auto statistics = rapidsmpf::Statistics::disabled();
    ctx_ = std::make_shared<rapidsmpf::streaming::Context>(options, comm_, br_, statistics);
}

Executor::~Executor() = default;

// TODO: Implement cudf to DuckDB DataChunk conversion
// This would use Arrow as an intermediate format

// Helper to move a TableChunk to device
static rapidsmpf::streaming::TableChunk to_device(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    rapidsmpf::streaming::TableChunk&& chunk
) {
    auto reservation = ctx->br()->reserve_device_memory_and_spill(
        chunk.make_available_cost(), true  // allow overbooking
    );
    return chunk.make_available(reservation);
}

// Sink coroutine that consumes and counts messages from a channel
static rapidsmpf::streaming::Node sink_and_count(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::atomic<size_t>& row_count
) {
    rapidsmpf::streaming::ShutdownAtExit closer{ch_in};
    
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        
        // Get the table chunk and move to device
        auto chunk = to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
        row_count += chunk.table_view().num_rows();
    }
}

std::vector<duckdb::DataChunk> Executor::Execute(
    std::unique_ptr<PhysicalOperator> plan
) {
    std::vector<duckdb::DataChunk> results;
    
    if (!plan) {
        return results;
    }
    
    // Build the streaming pipeline from the operator tree
    // We need to traverse the tree and create channels between operators
    
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
        return results;
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
    
    // Add a sink node to consume results
    std::atomic<size_t> row_count{0};
    nodes.push_back(sink_and_count(ctx_, channels[operators.size() - 1], row_count));
    
    // Execute the pipeline
    rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));
    
    // TODO: Collect actual results and convert to DuckDB DataChunks
    // For now, return empty results as placeholder
    
    return results;
}

}  // namespace rapidsmpf_duckdb


