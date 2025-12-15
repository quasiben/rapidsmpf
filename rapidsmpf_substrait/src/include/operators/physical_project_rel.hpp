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
 * @brief Physical operator for projecting/computing expressions.
 *
 * Corresponds to Substrait's ProjectRel. Computes new columns
 * from expressions over the input columns.
 */
class PhysicalProjectRel : public PhysicalOperator {
  public:
    /**
     * @brief Construct a projection operator.
     *
     * @param child The input operator.
     * @param expressions The projection expressions.
     */
    PhysicalProjectRel(
        std::unique_ptr<PhysicalOperator> child,
        std::vector<ExpressionInfo> expressions
    );

    [[nodiscard]] rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

    [[nodiscard]] std::string ToString() const override;

    /**
     * @brief Get the projection expressions.
     */
    [[nodiscard]] std::vector<ExpressionInfo> const& Expressions() const {
        return expressions_;
    }

  private:
    std::vector<ExpressionInfo> expressions_;
};

}  // namespace rapidsmpf_substrait

