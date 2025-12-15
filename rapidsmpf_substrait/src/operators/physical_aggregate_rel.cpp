/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_aggregate_rel.hpp"

#include <sstream>

// Undefine DEBUG to avoid conflict
#ifdef DEBUG
#undef DEBUG
#endif

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf_substrait {

PhysicalAggregateRel::PhysicalAggregateRel(
    std::unique_ptr<PhysicalOperator> child,
    std::vector<int32_t> group_by_columns,
    std::vector<AggregateInfo> aggregates
) : PhysicalOperator(
        PhysicalOperatorType::AGGREGATE,
        {},  // Output types will be derived
        1    // Aggregation typically produces few rows
    ),
    group_by_columns_(std::move(group_by_columns)),
    aggregates_(std::move(aggregates)) {
    
    // Derive output types: group-by columns + aggregate outputs
    auto const& child_types = child->OutputTypes();
    for (auto col_idx : group_by_columns_) {
        if (col_idx >= 0 && static_cast<size_t>(col_idx) < child_types.size()) {
            output_types_.push_back(child_types[col_idx]);
        }
    }
    for (auto const& agg : aggregates_) {
        output_types_.push_back(agg.output_type);
    }
    
    children_.push_back(std::move(child));
}

// Map to reduce aggregation
static std::unique_ptr<cudf::reduce_aggregation> MapReduceAggregation(std::string const& name) {
    if (name == "sum" || name == "add") {
        return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    } else if (name == "avg" || name == "mean") {
        return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
    } else if (name == "min") {
        return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    } else if (name == "max") {
        return cudf::make_max_aggregation<cudf::reduce_aggregation>();
    }
    return nullptr;
}

rapidsmpf::streaming::Node PhysicalAggregateRel::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    auto group_by_columns = group_by_columns_;
    auto aggregates = aggregates_;

    // For ungrouped aggregation, we accumulate across all chunks
    if (group_by_columns.empty()) {
        return [](
            std::shared_ptr<rapidsmpf::streaming::Context> ctx,
            std::shared_ptr<rapidsmpf::streaming::Channel> input,
            std::shared_ptr<rapidsmpf::streaming::Channel> output,
            std::vector<AggregateInfo> aggregates
        ) -> rapidsmpf::streaming::Node {
            rapidsmpf::streaming::ShutdownAtExit shutdown_guard(output);

            // Accumulators for streaming aggregation
            std::vector<double> sums(aggregates.size(), 0.0);
            std::vector<int64_t> counts(aggregates.size(), 0);
            std::vector<double> mins(aggregates.size(), std::numeric_limits<double>::max());
            std::vector<double> maxs(aggregates.size(), std::numeric_limits<double>::lowest());

            rmm::cuda_stream_view last_stream;
            rmm::device_async_resource_ref last_mr = ctx->br()->device_mr();

            // Process all chunks
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

                auto table_view = chunk->table_view();
                last_stream = chunk->stream();
                last_mr = ctx->br()->device_mr();

                // Process each aggregate
                for (size_t i = 0; i < aggregates.size(); i++) {
                    auto const& agg = aggregates[i];
                    
                    if (agg.function_name == "count" || agg.function_name == "count_star") {
                        counts[i] += table_view.num_rows();
                    } else if (!agg.argument_indices.empty()) {
                        auto col_idx = agg.argument_indices[0];
                        if (col_idx >= 0 && col_idx < table_view.num_columns()) {
                            auto const& col = table_view.column(col_idx);
                            
                            // Use cudf::reduce for the aggregation
                            auto reduce_agg = MapReduceAggregation(agg.function_name);
                            if (reduce_agg) {
                                auto result = cudf::reduce(col, *reduce_agg, cudf::data_type{cudf::type_id::FLOAT64}, last_stream, last_mr);
                                auto* scalar = static_cast<cudf::numeric_scalar<double>*>(result.get());
                                if (scalar->is_valid()) {
                                    double val = scalar->value(last_stream);
                                    if (agg.function_name == "min") {
                                        mins[i] = std::min(mins[i], val);
                                    } else if (agg.function_name == "max") {
                                        maxs[i] = std::max(maxs[i], val);
                                    } else {
                                        sums[i] += val;
                                    }
                                }
                                counts[i] += col.size() - col.null_count();
                            }
                        }
                    }
                }
            }

            // Compute final values and create result table
            std::vector<std::unique_ptr<cudf::column>> result_columns;
            
            for (size_t i = 0; i < aggregates.size(); i++) {
                auto const& agg = aggregates[i];
                double value = 0.0;
                
                if (agg.function_name == "count" || agg.function_name == "count_star") {
                    value = static_cast<double>(counts[i]);
                } else if (agg.function_name == "sum" || agg.function_name == "add") {
                    value = sums[i];
                } else if (agg.function_name == "avg" || agg.function_name == "mean") {
                    value = counts[i] > 0 ? sums[i] / static_cast<double>(counts[i]) : 0.0;
                } else if (agg.function_name == "min") {
                    value = mins[i];
                } else if (agg.function_name == "max") {
                    value = maxs[i];
                }

                // Create a single-element column with the result
                auto col = cudf::make_numeric_column(
                    cudf::data_type{cudf::type_id::FLOAT64},
                    1,
                    cudf::mask_state::UNALLOCATED,
                    last_stream,
                    last_mr
                );
                
                // Copy value to device
                cudaMemcpyAsync(
                    col->mutable_view().head<double>(),
                    &value,
                    sizeof(double),
                    cudaMemcpyHostToDevice,
                    last_stream.value()
                );
                
                result_columns.push_back(std::move(col));
            }

            // Send the result
            if (!result_columns.empty()) {
                auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
                auto result_chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(result_table),
                    last_stream
                );
                co_await output->send(rapidsmpf::streaming::to_message(0, std::move(result_chunk)));
            }
            co_await output->drain(ctx->executor());
        }(ctx, ch_in, ch_out, aggregates);
    }

    // TODO: Implement grouped aggregation using cudf::groupby
    throw std::runtime_error("GROUP BY aggregation not yet implemented");
}

std::string PhysicalAggregateRel::ToString() const {
    std::ostringstream oss;
    oss << "AGGREGATE(";
    if (!group_by_columns_.empty()) {
        oss << "group_by=" << group_by_columns_.size() << ", ";
    }
    oss << "aggs=" << aggregates_.size() << ")";
    return oss.str();
}

}  // namespace rapidsmpf_substrait
