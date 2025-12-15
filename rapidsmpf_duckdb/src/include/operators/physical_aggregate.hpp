/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../physical_operator.hpp"

#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"

namespace rapidsmpf_duckdb {

/**
 * @brief Physical operator for aggregation via rapidsmpf.
 * 
 * Converts from DuckDB's LogicalAggregate operator. Currently supports
 * ungrouped aggregations (no GROUP BY) with AVG, SUM, COUNT, MIN, MAX.
 */
class PhysicalAggregate : public PhysicalOperator {
  public:
    PhysicalAggregate(
        duckdb::LogicalAggregate& agg,
        std::unique_ptr<PhysicalOperator> child
    );

    [[nodiscard]] std::string GetName() const override {
        return groups_.empty() ? "RAPIDSMPF_UNGROUPED_AGGREGATE" : "RAPIDSMPF_GROUPED_AGGREGATE";
    }

    [[nodiscard]] bool IsSink() const noexcept override {
        return true;
    }

    [[nodiscard]] rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

    [[nodiscard]] bool IsUngrouped() const noexcept {
        return groups_.empty();
    }

    [[nodiscard]] std::vector<duckdb::unique_ptr<duckdb::Expression>> const& 
    GetGroups() const noexcept {
        return groups_;
    }

    [[nodiscard]] std::vector<duckdb::unique_ptr<duckdb::Expression>> const& 
    GetAggregates() const noexcept {
        return aggregates_;
    }

  private:
    std::vector<duckdb::unique_ptr<duckdb::Expression>> groups_;
    std::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates_;
};

}  // namespace rapidsmpf_duckdb






