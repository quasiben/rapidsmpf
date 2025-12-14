/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../physical_operator.hpp"

#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression.hpp"

namespace rapidsmpf_duckdb {

/**
 * @brief Physical operator for projections via rapidsmpf.
 * 
 * Converts from DuckDB's LogicalProjection operator. Handles column
 * selection and simple expressions.
 */
class PhysicalProjection : public PhysicalOperator {
  public:
    /**
     * @brief Construct a projection operator.
     * 
     * @param proj The LogicalProjection operator from DuckDB.
     * @param child The child physical operator to project from.
     */
    PhysicalProjection(
        duckdb::LogicalProjection& proj,
        std::unique_ptr<PhysicalOperator> child
    );

    [[nodiscard]] std::string GetName() const override {
        return "RAPIDSMPF_PROJECTION";
    }

    /**
     * @brief Build the streaming node for projection.
     * 
     * Creates a node that:
     * 1. Receives TableChunks from ch_in
     * 2. Selects/computes the projected columns
     * 3. Sends results to ch_out
     */
    [[nodiscard]] rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

    /**
     * @brief Get the projection expressions.
     */
    [[nodiscard]] std::vector<duckdb::unique_ptr<duckdb::Expression>> const& 
    GetExpressions() const noexcept {
        return expressions_;
    }

  private:
    std::vector<duckdb::unique_ptr<duckdb::Expression>> expressions_;
    std::vector<duckdb::idx_t> column_indices_;
};

}  // namespace rapidsmpf_duckdb




