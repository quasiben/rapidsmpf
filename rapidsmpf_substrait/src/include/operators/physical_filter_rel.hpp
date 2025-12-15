/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "../physical_operator.hpp"
#include "../substrait_plan_converter.hpp"

namespace rapidsmpf_substrait {

/**
 * @brief Physical operator for filtering rows.
 *
 * Corresponds to Substrait's FilterRel. Applies a boolean predicate
 * to filter rows from the input.
 */
class PhysicalFilterRel : public PhysicalOperator {
  public:
    /**
     * @brief Construct a filter operator.
     *
     * @param child The input operator.
     * @param condition The filter condition expression.
     */
    PhysicalFilterRel(
        std::unique_ptr<PhysicalOperator> child,
        ExpressionInfo condition
    );

    [[nodiscard]] rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

    [[nodiscard]] std::string ToString() const override;

    /**
     * @brief Get the filter condition.
     */
    [[nodiscard]] ExpressionInfo const& Condition() const { return condition_; }

  private:
    ExpressionInfo condition_;
};

}  // namespace rapidsmpf_substrait

