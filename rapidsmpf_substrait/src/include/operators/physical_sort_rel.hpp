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
 * @brief Physical operator for sorting.
 *
 * Corresponds to Substrait's SortRel. Sorts the input by
 * the specified columns.
 */
class PhysicalSortRel : public PhysicalOperator {
  public:
    /**
     * @brief Construct a sort operator.
     *
     * @param child The input operator.
     * @param sort_fields The fields to sort by.
     */
    PhysicalSortRel(
        std::unique_ptr<PhysicalOperator> child,
        std::vector<SortFieldInfo> sort_fields
    );

    [[nodiscard]] rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

    [[nodiscard]] std::string ToString() const override;

    /**
     * @brief Get the sort fields.
     */
    [[nodiscard]] std::vector<SortFieldInfo> const& SortFields() const {
        return sort_fields_;
    }

  private:
    std::vector<SortFieldInfo> sort_fields_;
};

}  // namespace rapidsmpf_substrait

