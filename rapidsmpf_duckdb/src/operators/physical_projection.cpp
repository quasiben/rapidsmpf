/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_projection.hpp"

#include <cudf/copying.hpp>

// Undefine DEBUG to avoid conflict with DuckDB's -DDEBUG flag and rapidsmpf's LOG_LEVEL::DEBUG
#ifdef DEBUG
#undef DEBUG
#endif

#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "duckdb/planner/expression/bound_columnref_expression.hpp"

namespace rapidsmpf_duckdb {

PhysicalProjection::PhysicalProjection(
    duckdb::LogicalProjection& proj,
    std::unique_ptr<PhysicalOperator> child
)
    : PhysicalOperator(
          duckdb::PhysicalOperatorType::PROJECTION,
          proj.types,
          proj.estimated_cardinality
      ) {
    // Copy the projection expressions
    for (auto& expr : proj.expressions) {
        expressions_.push_back(expr->Copy());
        
        // Extract column indices for simple column references
        if (expr->type == duckdb::ExpressionType::BOUND_COLUMN_REF) {
            auto& colref = expr->Cast<duckdb::BoundColumnRefExpression>();
            column_indices_.push_back(colref.binding.column_index);
        }
    }
    
    // Add the child
    AddChild(std::move(child));
}

rapidsmpf::streaming::Node PhysicalProjection::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    // Capture the column indices for use in the coroutine
    std::vector<duckdb::idx_t> indices = column_indices_;
    
    return [](std::shared_ptr<rapidsmpf::streaming::Context> ctx,
              std::shared_ptr<rapidsmpf::streaming::Channel> input,
              std::shared_ptr<rapidsmpf::streaming::Channel> output,
              std::vector<duckdb::idx_t> col_indices) 
        -> rapidsmpf::streaming::Node {
        
        rapidsmpf::streaming::ShutdownAtExit shutdown_guard(output);
        std::uint64_t seq = 0;
        
        while (true) {
            auto msg = co_await input->receive();
            if (msg.empty()) break;
            
            auto chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                msg.release<rapidsmpf::streaming::TableChunk>()
            );
            
            // Make it available on device if needed
            if (!chunk->is_available()) {
                auto [reservation, overbooking] = ctx->br()->reserve(
                    rapidsmpf::MemoryType::DEVICE,
                    chunk->make_available_cost(),
                    true  // allow overbooking
                );
                *chunk = chunk->make_available(reservation);
            }
            
            auto tbl_view = chunk->table_view();
            auto stream = chunk->stream();
            auto* mr = rmm::mr::get_current_device_resource();
            
            // Select only the requested columns
            if (!col_indices.empty()) {
                std::vector<cudf::size_type> cudf_indices;
                for (auto idx : col_indices) {
                    cudf_indices.push_back(static_cast<cudf::size_type>(idx));
                }
                
                // Create a new table with selected columns
                std::vector<cudf::column_view> selected_cols;
                for (auto idx : cudf_indices) {
                    if (idx < tbl_view.num_columns()) {
                        selected_cols.push_back(tbl_view.column(idx));
                    }
                }
                
                cudf::table_view projected_view(selected_cols);
                auto projected_table = std::make_unique<cudf::table>(projected_view, stream, mr);
                
                auto new_chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(projected_table),
                    stream
                );
                
                co_await output->send(rapidsmpf::streaming::to_message(seq++, std::move(new_chunk)));
            } else {
                // No projection needed, pass through
                co_await output->send(rapidsmpf::streaming::to_message(seq++, std::move(chunk)));
            }
        }
    }(ctx, ch_in, ch_out, std::move(indices));
}

}  // namespace rapidsmpf_duckdb



