/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_filter.hpp"

#include <memory>
#include <vector>

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/std/chrono>

// Undefine DEBUG to avoid conflict with DuckDB's -DDEBUG flag and rapidsmpf's LOG_LEVEL::DEBUG
#ifdef DEBUG
#undef DEBUG
#endif

#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace rapidsmpf_duckdb {

/**
 * @brief Helper class to convert DuckDB expressions to cudf AST expressions.
 * 
 * Manages the lifetime of all intermediate objects (scalars, column references,
 * literals, operations) needed to build the cudf AST expression tree.
 */
class AstExpressionConverter {
  public:
    explicit AstExpressionConverter(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
        : stream_(stream), mr_(mr) {}

    /**
     * @brief Convert a DuckDB expression to a cudf AST expression.
     * 
     * @param expr The DuckDB expression to convert.
     * @return Reference to the root cudf AST expression.
     */
    cudf::ast::expression& Convert(duckdb::Expression& expr) {
        return *ConvertImpl(expr);
    }

  private:
    rmm::cuda_stream_view stream_;
    rmm::device_async_resource_ref mr_;
    
    // Storage for owned objects - keeps them alive as long as the converter exists
    std::vector<std::unique_ptr<cudf::scalar>> scalars_;
    std::vector<std::unique_ptr<cudf::ast::literal>> literals_;
    std::vector<std::unique_ptr<cudf::ast::column_reference>> column_refs_;
    std::vector<std::unique_ptr<cudf::ast::operation>> operations_;

    cudf::ast::expression* ConvertImpl(duckdb::Expression& expr) {
        switch (expr.type) {
            case duckdb::ExpressionType::COMPARE_EQUAL:
            case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            case duckdb::ExpressionType::COMPARE_LESSTHAN:
            case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                return ConvertComparison(expr);

            case duckdb::ExpressionType::CONJUNCTION_AND:
            case duckdb::ExpressionType::CONJUNCTION_OR:
                return ConvertConjunction(expr);

            case duckdb::ExpressionType::BOUND_COLUMN_REF:
                return ConvertColumnRef(expr);

            case duckdb::ExpressionType::BOUND_REF:
                return ConvertBoundRef(expr);

            case duckdb::ExpressionType::VALUE_CONSTANT:
                return ConvertConstant(expr);

            case duckdb::ExpressionType::OPERATOR_CAST:
                return ConvertCast(expr);

            default:
                throw std::runtime_error(
                    "Unsupported expression type in filter: " + 
                    duckdb::ExpressionTypeToString(expr.type)
                );
        }
    }

    cudf::ast::expression* ConvertCast(duckdb::Expression& expr) {
        // For CAST, we need to handle the target type appropriately
        // cudf AST doesn't have explicit cast nodes, so we need to create 
        // properly typed literals when the child is a constant
        auto& cast_expr = expr.Cast<duckdb::BoundCastExpression>();
        auto target_type = cast_expr.return_type;
        
        // If the child is a constant, create a literal with the target type
        if (cast_expr.child->type == duckdb::ExpressionType::VALUE_CONSTANT) {
            auto& constant = cast_expr.child->Cast<duckdb::BoundConstantExpression>();
            
            // Get the value and manually cast it to the target type
            auto value = constant.value.DefaultCastAs(target_type);
            return ConvertConstantValue(value);
        }
        
        // For column references or complex expressions, just pass through
        // cudf will handle type matching at runtime
        return ConvertImpl(*cast_expr.child);
    }
    
    cudf::ast::expression* ConvertConstantValue(duckdb::Value& value) {
        switch (value.type().id()) {
            case duckdb::LogicalTypeId::BOOLEAN: {
                auto scalar = std::make_unique<cudf::numeric_scalar<bool>>(
                    value.GetValue<bool>(), true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            case duckdb::LogicalTypeId::TINYINT: {
                auto scalar = std::make_unique<cudf::numeric_scalar<int8_t>>(
                    value.GetValue<int8_t>(), true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            case duckdb::LogicalTypeId::SMALLINT: {
                auto scalar = std::make_unique<cudf::numeric_scalar<int16_t>>(
                    value.GetValue<int16_t>(), true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            case duckdb::LogicalTypeId::INTEGER: {
                auto scalar = std::make_unique<cudf::numeric_scalar<int32_t>>(
                    value.GetValue<int32_t>(), true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            case duckdb::LogicalTypeId::BIGINT: {
                auto scalar = std::make_unique<cudf::numeric_scalar<int64_t>>(
                    value.GetValue<int64_t>(), true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            case duckdb::LogicalTypeId::FLOAT: {
                auto scalar = std::make_unique<cudf::numeric_scalar<float>>(
                    value.GetValue<float>(), true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            case duckdb::LogicalTypeId::DOUBLE: {
                auto scalar = std::make_unique<cudf::numeric_scalar<double>>(
                    value.GetValue<double>(), true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            case duckdb::LogicalTypeId::DECIMAL: {
                auto dbl_val = value.GetValue<double>();
                auto scalar = std::make_unique<cudf::numeric_scalar<double>>(
                    dbl_val, true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            case duckdb::LogicalTypeId::VARCHAR: {
                // String literal - create a cudf::string_scalar
                auto str_val = value.GetValue<std::string>();
                auto scalar = std::make_unique<cudf::string_scalar>(
                    str_val, true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            case duckdb::LogicalTypeId::DATE: {
                // Date literal - convert to cudf::timestamp_scalar<timestamp_D>
                // DuckDB stores dates as days since epoch
                auto date_val = value.GetValue<duckdb::date_t>();
                auto days_since_epoch = date_val.days;
                
                // Create a timestamp scalar with day precision
                using timestamp_type = cudf::timestamp_D;
                auto scalar = std::make_unique<cudf::timestamp_scalar<timestamp_type>>(
                    timestamp_type::duration{days_since_epoch}, true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            case duckdb::LogicalTypeId::TIMESTAMP:
            case duckdb::LogicalTypeId::TIMESTAMP_TZ: {
                // Timestamp literal - convert to cudf::timestamp_scalar<timestamp_us>
                // DuckDB stores timestamps as microseconds since epoch
                auto ts_val = value.GetValue<duckdb::timestamp_t>();
                
                using timestamp_type = cudf::timestamp_us;
                auto scalar = std::make_unique<cudf::timestamp_scalar<timestamp_type>>(
                    timestamp_type::duration{ts_val.value}, true, stream_, mr_
                );
                literals_.push_back(std::make_unique<cudf::ast::literal>(*scalar));
                scalars_.push_back(std::move(scalar));
                return literals_.back().get();
            }
            default:
                throw std::runtime_error(
                    "Unsupported constant type in filter: " + 
                    value.type().ToString()
                );
        }
    }

    cudf::ast::expression* ConvertComparison(duckdb::Expression& expr) {
        auto& comp = expr.Cast<duckdb::BoundComparisonExpression>();
        
        auto* left = ConvertImpl(*comp.left);
        auto* right = ConvertImpl(*comp.right);
        
        cudf::ast::ast_operator op;
        switch (expr.type) {
            case duckdb::ExpressionType::COMPARE_EQUAL:
                op = cudf::ast::ast_operator::EQUAL;
                break;
            case duckdb::ExpressionType::COMPARE_NOTEQUAL:
                op = cudf::ast::ast_operator::NOT_EQUAL;
                break;
            case duckdb::ExpressionType::COMPARE_LESSTHAN:
                op = cudf::ast::ast_operator::LESS;
                break;
            case duckdb::ExpressionType::COMPARE_GREATERTHAN:
                op = cudf::ast::ast_operator::GREATER;
                break;
            case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
                op = cudf::ast::ast_operator::LESS_EQUAL;
                break;
            case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                op = cudf::ast::ast_operator::GREATER_EQUAL;
                break;
            default:
                throw std::runtime_error("Unexpected comparison type");
        }
        
        operations_.push_back(std::make_unique<cudf::ast::operation>(op, *left, *right));
        return operations_.back().get();
    }

    cudf::ast::expression* ConvertConjunction(duckdb::Expression& expr) {
        auto& conj = expr.Cast<duckdb::BoundConjunctionExpression>();
        
        if (conj.children.empty()) {
            throw std::runtime_error("Empty conjunction expression");
        }
        
        // Convert first child
        auto* result = ConvertImpl(*conj.children[0]);
        
        // Chain the rest with AND or OR
        cudf::ast::ast_operator op = (expr.type == duckdb::ExpressionType::CONJUNCTION_AND)
            ? cudf::ast::ast_operator::BITWISE_AND  // AND for boolean
            : cudf::ast::ast_operator::BITWISE_OR;  // OR for boolean
        
        for (size_t i = 1; i < conj.children.size(); i++) {
            auto* child = ConvertImpl(*conj.children[i]);
            operations_.push_back(std::make_unique<cudf::ast::operation>(op, *result, *child));
            result = operations_.back().get();
        }
        
        return result;
    }

    cudf::ast::expression* ConvertColumnRef(duckdb::Expression& expr) {
        auto& colref = expr.Cast<duckdb::BoundColumnRefExpression>();
        
        // Use column index from the binding
        auto col_idx = static_cast<cudf::size_type>(colref.binding.column_index);
        column_refs_.push_back(std::make_unique<cudf::ast::column_reference>(col_idx));
        return column_refs_.back().get();
    }

    cudf::ast::expression* ConvertBoundRef(duckdb::Expression& expr) {
        auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
        
        // Use the reference index
        auto col_idx = static_cast<cudf::size_type>(ref.index);
        column_refs_.push_back(std::make_unique<cudf::ast::column_reference>(col_idx));
        return column_refs_.back().get();
    }

    cudf::ast::expression* ConvertConstant(duckdb::Expression& expr) {
        auto& constant = expr.Cast<duckdb::BoundConstantExpression>();
        auto value = constant.value;  // Copy so we can modify if needed
        return ConvertConstantValue(value);
    }
};

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

rapidsmpf::streaming::Node PhysicalFilter::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    // Capture a copy of the expression for the coroutine
    auto expr_copy = expression_->Copy();
    
    return [](std::shared_ptr<rapidsmpf::streaming::Context> ctx,
              std::shared_ptr<rapidsmpf::streaming::Channel> input,
              std::shared_ptr<rapidsmpf::streaming::Channel> output,
              duckdb::unique_ptr<duckdb::Expression> filter_expr) 
        -> rapidsmpf::streaming::Node {
        
        // Shutdown both channels on exit (like Q9's filter_part pattern)
        rapidsmpf::streaming::ShutdownAtExit shutdown_guard{input, output};
        std::uint64_t seq = 0;
        
        while (true) {
            auto msg = co_await input->receive();
            if (msg.empty()) break;
            
            co_await ctx->executor()->schedule();
            
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
            
            auto tbl_view = chunk->table_view();
            auto stream = chunk->stream();
            auto mr = ctx->br()->device_mr();
            
            // Skip empty chunks
            if (tbl_view.num_rows() == 0) {
                co_await output->send(rapidsmpf::streaming::to_message(seq++, std::move(chunk)));
                continue;
            }
            
            // Convert the DuckDB expression to cudf AST
            AstExpressionConverter converter(stream, mr);
            auto& ast_expr = converter.Convert(*filter_expr);
            
            // Compute the boolean mask using cudf::compute_column
            auto mask_column = cudf::compute_column(tbl_view, ast_expr, stream, mr);
            
            // Apply the boolean mask to filter rows
            auto filtered_table = cudf::apply_boolean_mask(
                tbl_view, mask_column->view(), stream, mr
            );
            
            // Create a new chunk with the filtered data
            auto filtered_chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                std::move(filtered_table),
                stream
            );
            
            co_await output->send(rapidsmpf::streaming::to_message(seq++, std::move(filtered_chunk)));
        }
        
        // Drain the output channel before exiting
        co_await output->drain(ctx->executor());
    }(ctx, ch_in, ch_out, std::move(expr_copy));
}

}  // namespace rapidsmpf_duckdb
