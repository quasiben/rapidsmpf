/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/main/client_context.hpp"

#include "physical_operator.hpp"

namespace rapidsmpf_duckdb {

/**
 * @brief Converts DuckDB's LogicalOperator tree to rapidsmpf PhysicalOperator tree.
 * 
 * This class mirrors the pattern from sirius's GPUPhysicalPlanGenerator but
 * produces operators that execute via rapidsmpf's streaming framework.
 */
class PhysicalPlanGenerator {
  public:
    /**
     * @brief Construct a plan generator.
     * 
     * @param context DuckDB client context for accessing catalog, etc.
     */
    explicit PhysicalPlanGenerator(duckdb::ClientContext& context);

    /**
     * @brief Create a physical plan from a logical operator tree.
     * 
     * This resolves column bindings and types, then converts the tree.
     * 
     * @param op Root of the logical operator tree.
     * @return Root of the physical operator tree.
     */
    [[nodiscard]] std::unique_ptr<PhysicalOperator> CreatePlan(
        std::unique_ptr<duckdb::LogicalOperator> op
    );

  private:
    duckdb::ClientContext& context_;

    /**
     * @brief Convert a single logical operator to physical.
     * 
     * @param op The logical operator to convert.
     * @return The corresponding physical operator.
     */
    [[nodiscard]] std::unique_ptr<PhysicalOperator> CreatePlan(
        duckdb::LogicalOperator& op
    );

    // Specific operator conversions
    [[nodiscard]] std::unique_ptr<PhysicalOperator> CreatePlan(
        duckdb::LogicalGet& op
    );
    [[nodiscard]] std::unique_ptr<PhysicalOperator> CreatePlan(
        duckdb::LogicalFilter& op
    );
    [[nodiscard]] std::unique_ptr<PhysicalOperator> CreatePlan(
        duckdb::LogicalProjection& op
    );
    [[nodiscard]] std::unique_ptr<PhysicalOperator> CreatePlan(
        duckdb::LogicalAggregate& op
    );
};

}  // namespace rapidsmpf_duckdb




