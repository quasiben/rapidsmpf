/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_order.hpp"

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/concatenate.hpp>

// Undefine DEBUG to avoid conflict with DuckDB's -DDEBUG flag and rapidsmpf's LOG_LEVEL::DEBUG
#ifdef DEBUG
#undef DEBUG
#endif

#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "duckdb/planner/bound_query_node.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

namespace rapidsmpf_duckdb {

PhysicalOrder::PhysicalOrder(
    duckdb::LogicalOrder& op,
    std::unique_ptr<PhysicalOperator> child
) : PhysicalOperator(
        duckdb::PhysicalOperatorType::ORDER_BY,
        op.types,
        op.estimated_cardinality
    )
{
    AddChild(std::move(child));
    
    // Extract sort orders from the BoundOrderByNode
    for (auto& order : op.orders) {
        // Get column index from the expression
        if (order.expression->type == duckdb::ExpressionType::BOUND_COLUMN_REF) {
            auto& colref = order.expression->Cast<duckdb::BoundColumnRefExpression>();
            sort_columns_.push_back(static_cast<cudf::size_type>(colref.binding.column_index));
        } else {
            // For complex expressions, default to column 0
            // TODO: Handle complex sort expressions
            sort_columns_.push_back(0);
        }
        
        // Convert order type
        if (order.type == duckdb::OrderType::DESCENDING) {
            column_orders_.push_back(cudf::order::DESCENDING);
        } else {
            column_orders_.push_back(cudf::order::ASCENDING);
        }
        
        // Convert null order
        if (order.null_order == duckdb::OrderByNullType::NULLS_FIRST) {
            null_orders_.push_back(cudf::null_order::BEFORE);
        } else {
            null_orders_.push_back(cudf::null_order::AFTER);
        }
    }
}

rapidsmpf::streaming::Node PhysicalOrder::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    auto sort_cols = sort_columns_;
    auto col_orders = column_orders_;
    auto null_ords = null_orders_;
    
    return [](std::shared_ptr<rapidsmpf::streaming::Context> ctx,
              std::shared_ptr<rapidsmpf::streaming::Channel> input,
              std::shared_ptr<rapidsmpf::streaming::Channel> output,
              std::vector<cudf::size_type> sort_columns,
              std::vector<cudf::order> column_orders,
              std::vector<cudf::null_order> null_orders)
        -> rapidsmpf::streaming::Node {

        rapidsmpf::streaming::ShutdownAtExit shutdown_guard(input, output);
        
        // Collect all chunks (ORDER BY is a blocking operator)
        std::vector<std::unique_ptr<cudf::table>> tables;
        rmm::cuda_stream_view stream;
        
        while (true) {
            auto msg = co_await input->receive();
            if (msg.empty()) break;

            auto chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                msg.release<rapidsmpf::streaming::TableChunk>()
            );

            // Make chunk available on device
            if (!chunk->is_available()) {
                auto [reservation, overbooking] = ctx->br()->reserve(
                    rapidsmpf::MemoryType::DEVICE,
                    chunk->make_available_cost(),
                    true
                );
                *chunk = chunk->make_available(reservation);
            }

            stream = chunk->stream();
            auto mr = ctx->br()->device_mr();
            
            if (chunk->table_view().num_rows() > 0) {
                // Copy the table
                tables.push_back(std::make_unique<cudf::table>(
                    chunk->table_view(), stream, mr
                ));
            }
        }
        
        if (tables.empty()) {
            co_await output->drain(ctx->executor());
            co_return;
        }
        
        auto mr = ctx->br()->device_mr();
        
        // Concatenate all tables
        std::vector<cudf::table_view> table_views;
        for (auto& t : tables) {
            table_views.push_back(t->view());
        }
        
        auto combined_table = cudf::concatenate(table_views, stream, mr);
        tables.clear();  // Free memory
        
        // Select the sort key columns
        auto keys_view = combined_table->view().select(sort_columns);
        
        // Sort by key
        auto sorted_table = cudf::sort_by_key(
            combined_table->view(),
            keys_view,
            column_orders,
            null_orders,
            stream,
            mr
        );
        
        auto result_chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
            std::move(sorted_table),
            stream
        );
        
        co_await output->send(rapidsmpf::streaming::to_message(0, std::move(result_chunk)));
        co_await output->drain(ctx->executor());
    }(ctx, ch_in, ch_out, sort_cols, col_orders, null_ords);
}

}  // namespace rapidsmpf_duckdb
