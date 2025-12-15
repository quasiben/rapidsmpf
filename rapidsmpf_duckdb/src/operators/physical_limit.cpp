/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_limit.hpp"

#include <cudf/copying.hpp>

// Undefine DEBUG to avoid conflict with DuckDB's -DDEBUG flag and rapidsmpf's LOG_LEVEL::DEBUG
#ifdef DEBUG
#undef DEBUG
#endif

#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "duckdb/planner/expression/bound_constant_expression.hpp"

namespace rapidsmpf_duckdb {

PhysicalLimit::PhysicalLimit(
    duckdb::LogicalLimit& op,
    std::unique_ptr<PhysicalOperator> child
) : PhysicalOperator(
        duckdb::PhysicalOperatorType::LIMIT,
        op.types,
        op.estimated_cardinality
    ),
    limit_count_(-1),  // -1 means no limit
    offset_count_(0)
{
    AddChild(std::move(child));
    // Extract limit value
    if (op.limit_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
        limit_count_ = static_cast<int64_t>(op.limit_val.GetConstantValue());
    } else if (op.limit_val.Type() == duckdb::LimitNodeType::EXPRESSION_VALUE) {
        // Try to evaluate the expression if it's a constant
        auto& expr = *op.limit_val.GetExpression();
        if (expr.type == duckdb::ExpressionType::VALUE_CONSTANT) {
            auto& const_expr = expr.Cast<duckdb::BoundConstantExpression>();
            limit_count_ = const_expr.value.GetValue<int64_t>();
        }
    }
    
    // Extract offset value
    if (op.offset_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
        offset_count_ = static_cast<int64_t>(op.offset_val.GetConstantValue());
    } else if (op.offset_val.Type() == duckdb::LimitNodeType::EXPRESSION_VALUE) {
        auto& expr = *op.offset_val.GetExpression();
        if (expr.type == duckdb::ExpressionType::VALUE_CONSTANT) {
            auto& const_expr = expr.Cast<duckdb::BoundConstantExpression>();
            offset_count_ = const_expr.value.GetValue<int64_t>();
        }
    }
}

rapidsmpf::streaming::Node PhysicalLimit::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    int64_t limit = limit_count_;
    int64_t offset = offset_count_;
    
    return [](std::shared_ptr<rapidsmpf::streaming::Context> ctx,
              std::shared_ptr<rapidsmpf::streaming::Channel> input,
              std::shared_ptr<rapidsmpf::streaming::Channel> output,
              int64_t limit_count,
              int64_t offset_count)
        -> rapidsmpf::streaming::Node {

        rapidsmpf::streaming::ShutdownAtExit shutdown_guard(input, output);
        
        int64_t rows_skipped = 0;
        int64_t rows_emitted = 0;
        std::uint64_t seq = 0;

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

            auto tbl_view = chunk->table_view();
            auto stream = chunk->stream();
            auto mr = ctx->br()->device_mr();
            
            int64_t chunk_rows = tbl_view.num_rows();
            
            // Handle offset: skip rows first
            int64_t skip_in_chunk = 0;
            if (rows_skipped < offset_count) {
                int64_t remaining_to_skip = offset_count - rows_skipped;
                skip_in_chunk = std::min(remaining_to_skip, chunk_rows);
                rows_skipped += skip_in_chunk;
            }
            
            // Calculate how many rows to take from this chunk
            int64_t available_rows = chunk_rows - skip_in_chunk;
            if (available_rows <= 0) {
                continue;  // Skip this entire chunk
            }
            
            int64_t rows_to_take = available_rows;
            if (limit_count >= 0) {
                int64_t remaining_limit = limit_count - rows_emitted;
                rows_to_take = std::min(available_rows, remaining_limit);
            }
            
            if (rows_to_take <= 0) {
                // We've reached the limit, shutdown input
                break;
            }
            
            // Slice the table to get the desired rows
            std::unique_ptr<cudf::table> result_table;
            if (skip_in_chunk == 0 && rows_to_take == chunk_rows) {
                // Take the whole chunk
                result_table = std::make_unique<cudf::table>(tbl_view, stream, mr);
            } else {
                // Slice the chunk
                auto sliced_view = cudf::slice(
                    tbl_view, 
                    {static_cast<cudf::size_type>(skip_in_chunk), 
                     static_cast<cudf::size_type>(skip_in_chunk + rows_to_take)}
                )[0];
                result_table = std::make_unique<cudf::table>(sliced_view, stream, mr);
            }
            
            rows_emitted += rows_to_take;
            
            auto result_chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                std::move(result_table),
                stream
            );
            
            co_await output->send(rapidsmpf::streaming::to_message(seq++, std::move(result_chunk)));
            
            // Check if we've emitted enough rows
            if (limit_count >= 0 && rows_emitted >= limit_count) {
                break;
            }
        }
        
        co_await output->drain(ctx->executor());
    }(ctx, ch_in, ch_out, limit, offset);
}

}  // namespace rapidsmpf_duckdb

