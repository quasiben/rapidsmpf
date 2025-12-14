/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_filter.hpp"

#include <cudf/copying.hpp>
#include <cudf/ast/expressions.hpp>

// Undefine DEBUG to avoid conflict with DuckDB's -DDEBUG flag and rapidsmpf's LOG_LEVEL::DEBUG
#ifdef DEBUG
#undef DEBUG
#endif

#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf_duckdb {

PhysicalFilter::PhysicalFilter(
    duckdb::LogicalFilter& filter,
    std::unique_ptr<PhysicalOperator> child
)
    : PhysicalOperator(
          duckdb::PhysicalOperatorType::FILTER,
          child->GetTypes(),
          filter.estimated_cardinality
      ) {
    // Copy the filter expression
    expression_ = filter.expressions[0]->Copy();
    
    // Add the child
    AddChild(std::move(child));
}

/**
 * @brief Convert a DuckDB expression to a cudf AST expression.
 * 
 * This is a simplified converter for common filter expressions.
 * TODO: Implement full expression conversion.
 */
static std::unique_ptr<cudf::ast::expression> ConvertExpression(
    duckdb::Expression& expr,
    std::vector<std::unique_ptr<cudf::ast::literal>>& literals
) {
    // TODO: Implement expression conversion from DuckDB to cudf AST
    // This would handle:
    // - ComparisonExpression (=, <, >, <=, >=, !=)
    // - ConjunctionExpression (AND, OR)
    // - BoundColumnRefExpression (column references)
    // - ConstantExpression (literals)
    throw std::runtime_error(
        "Expression conversion from DuckDB to cudf AST is not yet implemented. "
        "Filter pushdown is stubbed out for future implementation."
    );
}

rapidsmpf::streaming::Node PhysicalFilter::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    // Create a filter node that:
    // 1. Receives TableChunks from ch_in
    // 2. Applies the filter predicate
    // 3. Sends filtered results to ch_out
    
    // For now, we create a simple pass-through node as a placeholder
    // TODO: Implement proper filter expression evaluation
    
    return [](std::shared_ptr<rapidsmpf::streaming::Context> ctx,
              std::shared_ptr<rapidsmpf::streaming::Channel> input,
              std::shared_ptr<rapidsmpf::streaming::Channel> output) 
        -> rapidsmpf::streaming::Node {
        
        rapidsmpf::streaming::ShutdownAtExit shutdown_guard(output);
        std::uint64_t seq = 0;
        
        while (true) {
            auto msg = co_await input->receive();
            if (msg.empty()) break;
            
            // Get the table chunk
            auto chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                msg.release<rapidsmpf::streaming::TableChunk>()
            );
            
            // Check if available, if not make it available
            if (!chunk->is_available()) {
                auto [reservation, overbooking] = ctx->br()->reserve(
                    rapidsmpf::MemoryType::DEVICE,
                    chunk->make_available_cost(),
                    true  // allow overbooking
                );
                *chunk = chunk->make_available(reservation);
            }
            
            // TODO: Apply the filter expression here
            // For now, pass through unchanged
            // cudf::ast::compute_column would be used here
            
            co_await output->send(rapidsmpf::streaming::to_message(seq++, std::move(chunk)));
        }
    }(ctx, ch_in, ch_out);
}

}  // namespace rapidsmpf_duckdb



