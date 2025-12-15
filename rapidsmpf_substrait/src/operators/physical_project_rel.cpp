/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_project_rel.hpp"

#include <sstream>

// Undefine DEBUG to avoid conflict
#ifdef DEBUG
#undef DEBUG
#endif

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf_substrait {

PhysicalProjectRel::PhysicalProjectRel(
    std::unique_ptr<PhysicalOperator> child,
    std::vector<ExpressionInfo> expressions
) : PhysicalOperator(
        PhysicalOperatorType::PROJECT,
        {},  // Output types will be derived from expressions
        child->EstimatedCardinality()
    ),
    expressions_(std::move(expressions)) {
    
    // Derive output types from expressions
    for (auto const& expr : expressions_) {
        output_types_.push_back(expr.output_type);
    }
    
    children_.push_back(std::move(child));
}

rapidsmpf::streaming::Node PhysicalProjectRel::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    // Capture expressions by value for the coroutine
    auto expressions = expressions_;

    return [](
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> input,
        std::shared_ptr<rapidsmpf::streaming::Channel> output,
        std::vector<ExpressionInfo> expressions
    ) -> rapidsmpf::streaming::Node {
        rapidsmpf::streaming::ShutdownAtExit shutdown_guard(output);
        std::uint64_t seq = 0;

        while (true) {
            auto msg = co_await input->receive();
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

            auto table_view = chunk->table_view();
            auto stream = chunk->stream();
            auto mr = ctx->br()->device_mr();
            
            // Build output columns based on expressions
            std::vector<cudf::column_view> output_columns;
            
            for (auto const& expr : expressions) {
                if (expr.type == ExpressionInfo::Type::FIELD_REFERENCE) {
                    // Simple column reference - just select the column
                    if (expr.field_index >= 0 && 
                        static_cast<size_t>(expr.field_index) < static_cast<size_t>(table_view.num_columns())) {
                        output_columns.push_back(table_view.column(expr.field_index));
                    }
                }
                // TODO: Handle computed expressions (arithmetic, functions, etc.)
            }

            if (!output_columns.empty()) {
                // Create a new table view with the projected columns
                cudf::table_view projected_view(output_columns);
                auto projected_table = std::make_unique<cudf::table>(projected_view, stream, mr);
                
                auto new_chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(projected_table),
                    stream
                );
                co_await output->send(rapidsmpf::streaming::to_message(seq++, std::move(new_chunk)));
            }
        }
        co_await output->drain(ctx->executor());
    }(ctx, ch_in, ch_out, expressions);
}

std::string PhysicalProjectRel::ToString() const {
    std::ostringstream oss;
    oss << "PROJECT(exprs=" << expressions_.size() << ")";
    return oss.str();
}

}  // namespace rapidsmpf_substrait
