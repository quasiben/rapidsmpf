/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_sort_rel.hpp"

#include <sstream>

// Undefine DEBUG to avoid conflict
#ifdef DEBUG
#undef DEBUG
#endif

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>

#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf_substrait {

PhysicalSortRel::PhysicalSortRel(
    std::unique_ptr<PhysicalOperator> child,
    std::vector<SortFieldInfo> sort_fields
) : PhysicalOperator(
        PhysicalOperatorType::SORT,
        child->OutputTypes(),
        child->EstimatedCardinality()
    ),
    sort_fields_(std::move(sort_fields)) {
    children_.push_back(std::move(child));
}

rapidsmpf::streaming::Node PhysicalSortRel::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    auto sort_fields = sort_fields_;

    return [](
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> input,
        std::shared_ptr<rapidsmpf::streaming::Channel> output,
        std::vector<SortFieldInfo> sort_fields
    ) -> rapidsmpf::streaming::Node {
        rapidsmpf::streaming::ShutdownAtExit shutdown_guard(output);

        // Collect all chunks (sorting requires all data)
        std::vector<std::unique_ptr<cudf::table>> tables;
        rmm::cuda_stream_view last_stream;
        rmm::device_async_resource_ref last_mr = ctx->br()->device_mr();

        while (true) {
            auto msg = co_await input->receive();
            if (msg.empty()) {
                break;
            }
            co_await ctx->executor()->schedule();

            auto chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                msg.release<rapidsmpf::streaming::TableChunk>()
            );

            if (!chunk->is_available()) {
                auto [reservation, overbooking] = ctx->br()->reserve(
                    rapidsmpf::MemoryType::DEVICE,
                    chunk->make_available_cost(),
                    true
                );
                *chunk = chunk->make_available(reservation);
            }

            last_stream = chunk->stream();
            last_mr = ctx->br()->device_mr();
            tables.push_back(std::make_unique<cudf::table>(chunk->table_view(), last_stream, last_mr));
        }

        if (tables.empty()) {
            co_await output->drain(ctx->executor());
            co_return;
        }

        // Concatenate all tables
        std::vector<cudf::table_view> views;
        for (auto const& t : tables) {
            views.push_back(t->view());
        }
        auto combined = cudf::concatenate(views, last_stream, last_mr);

        // Build sort column indices and orders
        std::vector<cudf::order> column_orders;
        std::vector<cudf::null_order> null_orders;

        for (auto const& sf : sort_fields) {
            column_orders.push_back(
                sf.ascending ? cudf::order::ASCENDING : cudf::order::DESCENDING
            );
            null_orders.push_back(
                sf.nulls_first ? cudf::null_order::BEFORE : cudf::null_order::AFTER
            );
        }

        // Sort the table
        auto sorted_indices = cudf::sorted_order(
            combined->view(),
            column_orders,
            null_orders,
            last_stream,
            last_mr
        );
        auto sorted_table = cudf::gather(
            combined->view(), 
            sorted_indices->view(),
            cudf::out_of_bounds_policy::DONT_CHECK,
            last_stream,
            last_mr
        );

        // Send the sorted result
        auto result_chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
            std::move(sorted_table),
            last_stream
        );
        co_await output->send(rapidsmpf::streaming::to_message(0, std::move(result_chunk)));
        co_await output->drain(ctx->executor());
    }(ctx, ch_in, ch_out, sort_fields);
}

std::string PhysicalSortRel::ToString() const {
    std::ostringstream oss;
    oss << "SORT(fields=" << sort_fields_.size() << ")";
    return oss.str();
}

}  // namespace rapidsmpf_substrait
