/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_aggregate.hpp"

#include <cudf/reduction.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>

// Undefine DEBUG to avoid conflict with DuckDB's -DDEBUG flag and rapidsmpf's LOG_LEVEL::DEBUG
#ifdef DEBUG
#undef DEBUG
#endif

#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

namespace rapidsmpf_duckdb {

PhysicalAggregate::PhysicalAggregate(
    duckdb::LogicalAggregate& agg,
    std::unique_ptr<PhysicalOperator> child
)
    : PhysicalOperator(
          duckdb::PhysicalOperatorType::UNGROUPED_AGGREGATE,
          agg.types,
          1  // Ungrouped aggregate produces 1 row
      ) {
    // Copy the group expressions (for future GROUP BY support)
    for (auto& expr : agg.groups) {
        groups_.push_back(expr->Copy());
    }
    
    // Copy the aggregate expressions
    for (auto& expr : agg.expressions) {
        aggregates_.push_back(expr->Copy());
    }
    
    // Add the child
    AddChild(std::move(child));
}

/**
 * @brief Get the cudf aggregation kind for a DuckDB aggregate function.
 */
static cudf::aggregation::Kind GetAggregationKind(std::string const& func_name) {
    if (func_name == "sum") {
        return cudf::aggregation::Kind::SUM;
    } else if (func_name == "avg" || func_name == "mean") {
        return cudf::aggregation::Kind::MEAN;
    } else if (func_name == "count" || func_name == "count_star") {
        return cudf::aggregation::Kind::COUNT_ALL;
    } else if (func_name == "min") {
        return cudf::aggregation::Kind::MIN;
    } else if (func_name == "max") {
        return cudf::aggregation::Kind::MAX;
    } else {
        throw std::runtime_error("Unsupported aggregation function: " + func_name);
    }
}

rapidsmpf::streaming::Node PhysicalAggregate::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    // For ungrouped aggregation, we need to:
    // 1. Collect all chunks
    // 2. Compute running aggregates (for SUM, COUNT, etc.) or collect all data (for MEAN)
    // 3. Finalize and output a single result row
    
    // Extract aggregation info for the coroutine
    struct AggInfo {
        cudf::aggregation::Kind kind;
        cudf::size_type column_idx;
    };
    
    std::vector<AggInfo> agg_infos;
    for (auto const& expr : aggregates_) {
        if (expr->type == duckdb::ExpressionType::BOUND_AGGREGATE) {
            auto& agg_expr = expr->Cast<duckdb::BoundAggregateExpression>();
            AggInfo info;
            info.kind = GetAggregationKind(agg_expr.function.name);
            
            // Get the column index from the first child expression
            if (!agg_expr.children.empty() && 
                agg_expr.children[0]->type == duckdb::ExpressionType::BOUND_COLUMN_REF) {
                auto& colref = agg_expr.children[0]->Cast<duckdb::BoundColumnRefExpression>();
                info.column_idx = static_cast<cudf::size_type>(colref.binding.column_index);
            } else {
                info.column_idx = 0;  // COUNT(*) doesn't need a column
            }
            
            agg_infos.push_back(info);
        }
    }
    
    return [](std::shared_ptr<rapidsmpf::streaming::Context> ctx,
              std::shared_ptr<rapidsmpf::streaming::Channel> input,
              std::shared_ptr<rapidsmpf::streaming::Channel> output,
              std::vector<AggInfo> aggs)
        -> rapidsmpf::streaming::Node {
        
        rapidsmpf::streaming::ShutdownAtExit shutdown_guard(output);
        
        // Accumulators for streaming aggregation
        std::vector<double> sums(aggs.size(), 0.0);
        std::vector<int64_t> counts(aggs.size(), 0);
        std::vector<double> mins(aggs.size(), std::numeric_limits<double>::max());
        std::vector<double> maxs(aggs.size(), std::numeric_limits<double>::lowest());
        
        // Process all chunks
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
            
            // For each aggregation, update the running totals
            for (size_t i = 0; i < aggs.size(); i++) {
                auto const& agg = aggs[i];
                
                if (agg.kind == cudf::aggregation::Kind::COUNT_ALL) {
                    counts[i] += tbl_view.num_rows();
                } else if (agg.column_idx < tbl_view.num_columns()) {
                    auto col = tbl_view.column(agg.column_idx);
                    
                    // Compute the aggregation for this chunk
                    auto agg_obj = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
                    auto result = cudf::reduce(col, *agg_obj, col.type(), stream, mr);
                    
                    // For now, assume numeric type and add to sum
                    // TODO: Handle different aggregation types properly
                    if (auto* numeric = dynamic_cast<cudf::numeric_scalar<double>*>(result.get())) {
                        if (numeric->is_valid()) {
                            sums[i] += numeric->value();
                            counts[i] += col.size() - col.null_count();
                            
                            // Also track min/max
                            auto min_agg = cudf::make_min_aggregation<cudf::reduce_aggregation>();
                            auto min_result = cudf::reduce(col, *min_agg, col.type(), stream, mr);
                            if (auto* min_scalar = dynamic_cast<cudf::numeric_scalar<double>*>(min_result.get())) {
                                if (min_scalar->is_valid()) {
                                    mins[i] = std::min(mins[i], min_scalar->value());
                                }
                            }
                            
                            auto max_agg = cudf::make_max_aggregation<cudf::reduce_aggregation>();
                            auto max_result = cudf::reduce(col, *max_agg, col.type(), stream, mr);
                            if (auto* max_scalar = dynamic_cast<cudf::numeric_scalar<double>*>(max_result.get())) {
                                if (max_scalar->is_valid()) {
                                    maxs[i] = std::max(maxs[i], max_scalar->value());
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Build the final result
        std::vector<std::unique_ptr<cudf::column>> result_columns;
        rmm::cuda_stream_view stream = rmm::cuda_stream_default;
        auto* mr = rmm::mr::get_current_device_resource();
        
        for (size_t i = 0; i < aggs.size(); i++) {
            auto const& agg = aggs[i];
            double value = 0.0;
            
            switch (agg.kind) {
                case cudf::aggregation::Kind::SUM:
                    value = sums[i];
                    break;
                case cudf::aggregation::Kind::MEAN:
                    value = counts[i] > 0 ? sums[i] / static_cast<double>(counts[i]) : 0.0;
                    break;
                case cudf::aggregation::Kind::COUNT_ALL:
                    value = static_cast<double>(counts[i]);
                    break;
                case cudf::aggregation::Kind::MIN:
                    value = mins[i];
                    break;
                case cudf::aggregation::Kind::MAX:
                    value = maxs[i];
                    break;
                default:
                    throw std::runtime_error("Unsupported aggregation kind");
            }
            
            // Create a column with the single result value
            auto col = cudf::make_numeric_column(
                cudf::data_type{cudf::type_id::FLOAT64},
                1,
                cudf::mask_state::UNALLOCATED,
                stream,
                mr
            );
            
            // Set the value
            cudf::numeric_scalar<double> scalar(value, true, stream, mr);
            // TODO: Actually set the column value from the scalar
            
            result_columns.push_back(std::move(col));
        }
        
        // Create the result table
        auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
        auto result_chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
            std::move(result_table),
            stream
        );
        
        co_await output->send(rapidsmpf::streaming::to_message(0, std::move(result_chunk)));
    }(ctx, ch_in, ch_out, std::move(agg_infos));
}

}  // namespace rapidsmpf_duckdb



