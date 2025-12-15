/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file rapidsmpf_substrait.hpp
 * @brief Main header for rapidsmpf_substrait library.
 *
 * This library provides a Substrait interface for rapidsmpf, enabling
 * execution of Substrait query plans on GPU using RAPIDS cudf.
 *
 * ## Quick Start
 *
 * ```cpp
 * #include <rapidsmpf_substrait.hpp>
 *
 * // Parse a Substrait plan from JSON
 * auto plan = rapidsmpf_substrait::SubstraitPlanParser::ParseFromJson(json_str);
 *
 * // Create a parser and converter
 * rapidsmpf_substrait::SubstraitPlanParser parser(plan);
 * rapidsmpf_substrait::SubstraitPlanConverter converter(parser);
 *
 * // Convert to physical plan
 * auto physical_plan = converter.Convert();
 *
 * // Execute on GPU
 * rapidsmpf_substrait::Executor executor;
 * auto result = executor.Execute(std::move(physical_plan));
 *
 * std::cout << "Rows: " << result.rows_processed << std::endl;
 * std::cout << "Time: " << result.execution_time_ms << " ms" << std::endl;
 * ```
 *
 * ## Supported Operations
 *
 * | Substrait Relation | Physical Operator | Status |
 * |--------------------|-------------------|--------|
 * | ReadRel (parquet)  | PhysicalReadRel   | ✅ Supported |
 * | FilterRel          | PhysicalFilterRel | ✅ Supported |
 * | ProjectRel         | PhysicalProjectRel| ✅ Supported |
 * | AggregateRel       | PhysicalAggregateRel | ✅ Ungrouped only |
 * | SortRel            | PhysicalSortRel   | ✅ Supported |
 * | FetchRel           | PhysicalFetchRel  | ✅ Supported |
 * | JoinRel            | -                 | ❌ Not implemented |
 * | SetRel             | -                 | ❌ Not implemented |
 */

// Core components
#include "executor.hpp"
#include "physical_operator.hpp"
#include "substrait_plan_parser.hpp"
#include "substrait_plan_converter.hpp"

// Physical operators
#include "operators/physical_read_rel.hpp"
#include "operators/physical_filter_rel.hpp"
#include "operators/physical_project_rel.hpp"
#include "operators/physical_aggregate_rel.hpp"
#include "operators/physical_sort_rel.hpp"
#include "operators/physical_fetch_rel.hpp"

namespace rapidsmpf_substrait {

/**
 * @brief Execute a Substrait plan from JSON.
 *
 * Convenience function that parses, converts, and executes a Substrait plan.
 *
 * @param json The Substrait plan in JSON format.
 * @return The execution result.
 */
inline ExecutionResult ExecuteFromJson(std::string const& json) {
    auto plan = SubstraitPlanParser::ParseFromJson(json);
    SubstraitPlanParser parser(std::move(plan));
    SubstraitPlanConverter converter(parser);
    auto physical_plan = converter.Convert();
    
    Executor executor;
    return executor.Execute(std::move(physical_plan));
}

/**
 * @brief Execute a Substrait plan from binary protobuf.
 *
 * Convenience function that parses, converts, and executes a Substrait plan.
 *
 * @param data The Substrait plan in binary protobuf format.
 * @return The execution result.
 */
inline ExecutionResult ExecuteFromBinary(std::string const& data) {
    auto plan = SubstraitPlanParser::ParseFromBinary(data);
    SubstraitPlanParser parser(std::move(plan));
    SubstraitPlanConverter converter(parser);
    auto physical_plan = converter.Convert();
    
    Executor executor;
    return executor.Execute(std::move(physical_plan));
}

/**
 * @brief Execute a Substrait plan from a file.
 *
 * Convenience function that loads, parses, converts, and executes a Substrait plan.
 * The file format (JSON or binary) is auto-detected.
 *
 * @param file_path Path to the Substrait plan file.
 * @return The execution result.
 */
inline ExecutionResult ExecuteFromFile(std::string const& file_path) {
    auto plan = SubstraitPlanParser::ParseFromFile(file_path);
    SubstraitPlanParser parser(std::move(plan));
    SubstraitPlanConverter converter(parser);
    auto physical_plan = converter.Convert();
    
    Executor executor;
    return executor.Execute(std::move(physical_plan));
}

}  // namespace rapidsmpf_substrait

