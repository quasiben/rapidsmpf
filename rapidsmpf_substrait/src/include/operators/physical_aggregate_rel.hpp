/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

#include "../physical_operator.hpp"
#include "../substrait_plan_converter.hpp"

namespace rapidsmpf_substrait {

/**
 * @brief Physical operator for aggregation.
 *
 * Corresponds to Substrait's AggregateRel. Computes aggregate
 * functions over groups of rows.
 */
class PhysicalAggregateRel : public PhysicalOperator {
  public:
    /**
     * @brief Construct an aggregate operator.
     *
     * @param child The input operator.
     * @param group_by_columns Indices of columns to group by.
     * @param aggregates The aggregate functions to compute.
     */
    PhysicalAggregateRel(
        std::unique_ptr<PhysicalOperator> child,
        std::vector<int32_t> group_by_columns,
        std::vector<AggregateInfo> aggregates
    );

    [[nodiscard]] rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

    [[nodiscard]] std::string ToString() const override;

    /**
     * @brief Get the group-by column indices.
     */
    [[nodiscard]] std::vector<int32_t> const& GroupByColumns() const {
        return group_by_columns_;
    }

    /**
     * @brief Get the aggregate functions.
     */
    [[nodiscard]] std::vector<AggregateInfo> const& Aggregates() const {
        return aggregates_;
    }

    /**
     * @brief Check if this is an ungrouped aggregation.
     */
    [[nodiscard]] bool IsUngrouped() const { return group_by_columns_.empty(); }

  private:
    std::vector<int32_t> group_by_columns_;
    std::vector<AggregateInfo> aggregates_;
};

}  // namespace rapidsmpf_substrait

