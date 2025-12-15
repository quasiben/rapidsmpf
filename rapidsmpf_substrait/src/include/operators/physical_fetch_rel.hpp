/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "../physical_operator.hpp"

namespace rapidsmpf_substrait {

/**
 * @brief Physical operator for LIMIT/OFFSET.
 *
 * Corresponds to Substrait's FetchRel. Limits the number of
 * output rows with an optional offset.
 */
class PhysicalFetchRel : public PhysicalOperator {
  public:
    /**
     * @brief Construct a fetch (limit) operator.
     *
     * @param child The input operator.
     * @param offset Number of rows to skip.
     * @param count Maximum number of rows to return (-1 = unlimited).
     */
    PhysicalFetchRel(
        std::unique_ptr<PhysicalOperator> child,
        int64_t offset,
        int64_t count
    );

    [[nodiscard]] rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

    [[nodiscard]] std::string ToString() const override;

    /**
     * @brief Get the offset.
     */
    [[nodiscard]] int64_t Offset() const { return offset_; }

    /**
     * @brief Get the count (-1 = unlimited).
     */
    [[nodiscard]] int64_t Count() const { return count_; }

  private:
    int64_t offset_;
    int64_t count_;
};

}  // namespace rapidsmpf_substrait

