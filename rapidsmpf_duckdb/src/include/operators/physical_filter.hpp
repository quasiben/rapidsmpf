/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../physical_operator.hpp"

#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/expression.hpp"

namespace rapidsmpf_duckdb {

/**
 * @brief Physical operator for filtering via rapidsmpf.
 * 
 * Converts from DuckDB's LogicalFilter operator. The filter expression
 * is converted to a libcudf AST expression and applied via cudf::copy_if.
 */
class PhysicalFilter : public PhysicalOperator {
  public:
    /**
     * @brief Construct a filter operator.
     * 
     * @param filter The LogicalFilter operator from DuckDB.
     * @param child The child physical operator to filter.
     */
    PhysicalFilter(
        duckdb::LogicalFilter& filter,
        std::unique_ptr<PhysicalOperator> child
    );

    [[nodiscard]] std::string GetName() const override {
        return "RAPIDSMPF_FILTER";
    }

    /**
     * @brief Build the streaming node for filtering.
     * 
     * Creates a node that:
     * 1. Receives TableChunks from ch_in
     * 2. Applies the filter predicate using cudf::copy_if
     * 3. Sends filtered results to ch_out
     */
    [[nodiscard]] rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

    /**
     * @brief Get the filter expression.
     */
    [[nodiscard]] duckdb::Expression& GetExpression() const noexcept {
        return *expression_;
    }

  private:
    duckdb::unique_ptr<duckdb::Expression> expression_;
};

}  // namespace rapidsmpf_duckdb



