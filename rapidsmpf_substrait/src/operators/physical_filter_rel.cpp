/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_filter_rel.hpp"

#include <sstream>

// Undefine DEBUG to avoid conflict
#ifdef DEBUG
#undef DEBUG
#endif

#include <cudf/ast/expressions.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>

#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf_substrait {

PhysicalFilterRel::PhysicalFilterRel(
    std::unique_ptr<PhysicalOperator> child,
    ExpressionInfo condition
) : PhysicalOperator(
        PhysicalOperatorType::FILTER,
        child->OutputTypes(),
        child->EstimatedCardinality() / 2  // Assume ~50% selectivity
    ),
    condition_(std::move(condition)) {
    children_.push_back(std::move(child));
}

// Helper to build a cudf AST expression from our ExpressionInfo
static std::unique_ptr<cudf::ast::operation> BuildAstExpression(
    ExpressionInfo const& expr,
    std::vector<std::unique_ptr<cudf::ast::literal>>& literals
) {
    if (expr.type == ExpressionInfo::Type::SCALAR_FUNCTION) {
        // Map function names to AST operators
        cudf::ast::ast_operator op;
        
        if (expr.function_name == "equal" || expr.function_name == "eq") {
            op = cudf::ast::ast_operator::EQUAL;
        } else if (expr.function_name == "not_equal" || expr.function_name == "ne") {
            op = cudf::ast::ast_operator::NOT_EQUAL;
        } else if (expr.function_name == "lt" || expr.function_name == "less_than") {
            op = cudf::ast::ast_operator::LESS;
        } else if (expr.function_name == "lte" || expr.function_name == "less_than_or_equal") {
            op = cudf::ast::ast_operator::LESS_EQUAL;
        } else if (expr.function_name == "gt" || expr.function_name == "greater_than") {
            op = cudf::ast::ast_operator::GREATER;
        } else if (expr.function_name == "gte" || expr.function_name == "greater_than_or_equal") {
            op = cudf::ast::ast_operator::GREATER_EQUAL;
        } else if (expr.function_name == "and") {
            op = cudf::ast::ast_operator::LOGICAL_AND;
        } else if (expr.function_name == "or") {
            op = cudf::ast::ast_operator::LOGICAL_OR;
        } else {
            // For unsupported functions, return nullptr
            return nullptr;
        }

        // Build operands (simplified: only handle binary operations with column and literal)
        if (expr.arguments.size() == 2) {
            auto const& left = expr.arguments[0];
            auto const& right = expr.arguments[1];

            // Case: column op literal
            if (left.type == ExpressionInfo::Type::FIELD_REFERENCE &&
                right.type == ExpressionInfo::Type::LITERAL) {
                
                auto col_ref = cudf::ast::column_reference(left.field_index);
                
                // Create literal based on type
                std::unique_ptr<cudf::ast::literal> lit;
                switch (right.output_type.id()) {
                    case cudf::type_id::INT32: {
                        auto scalar = cudf::numeric_scalar<int32_t>(std::stoi(right.literal_value));
                        literals.push_back(std::make_unique<cudf::ast::literal>(scalar));
                        lit = std::move(literals.back());
                        break;
                    }
                    case cudf::type_id::INT64: {
                        auto scalar = cudf::numeric_scalar<int64_t>(std::stoll(right.literal_value));
                        literals.push_back(std::make_unique<cudf::ast::literal>(scalar));
                        lit = std::move(literals.back());
                        break;
                    }
                    case cudf::type_id::FLOAT64: {
                        auto scalar = cudf::numeric_scalar<double>(std::stod(right.literal_value));
                        literals.push_back(std::make_unique<cudf::ast::literal>(scalar));
                        lit = std::move(literals.back());
                        break;
                    }
                    default:
                        return nullptr;
                }
                
                return std::make_unique<cudf::ast::operation>(op, col_ref, *literals.back());
            }
        }
    }
    
    return nullptr;
}

rapidsmpf::streaming::Node PhysicalFilterRel::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    // Capture condition by value for the coroutine
    auto condition = condition_;

    return [](
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> input,
        std::shared_ptr<rapidsmpf::streaming::Channel> output,
        ExpressionInfo condition
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
            
            // Try to build and apply AST filter
            std::vector<std::unique_ptr<cudf::ast::literal>> literals;
            auto ast_expr = BuildAstExpression(condition, literals);
            
            std::unique_ptr<cudf::table> filtered_table;
            if (ast_expr) {
                // Compute boolean mask using AST
                auto mask = cudf::compute_column(table_view, *ast_expr, stream, mr);
                filtered_table = cudf::apply_boolean_mask(table_view, mask->view(), stream, mr);
            } else {
                // No valid AST expression - pass through (TODO: implement more cases)
                filtered_table = std::make_unique<cudf::table>(table_view, stream, mr);
            }

            // Only send if there are rows
            if (filtered_table->num_rows() > 0) {
                auto new_chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(filtered_table),
                    stream
                );
                co_await output->send(rapidsmpf::streaming::to_message(seq++, std::move(new_chunk)));
            }
        }
        co_await output->drain(ctx->executor());
    }(ctx, ch_in, ch_out, condition);
}

std::string PhysicalFilterRel::ToString() const {
    std::ostringstream oss;
    oss << "FILTER(";
    if (condition_.type == ExpressionInfo::Type::SCALAR_FUNCTION) {
        oss << condition_.function_name;
    } else {
        oss << "expr";
    }
    oss << ")";
    return oss.str();
}

}  // namespace rapidsmpf_substrait
