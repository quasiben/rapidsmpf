/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../physical_operator.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"

namespace rapidsmpf_duckdb {

/**
 * @brief Physical operator for LIMIT/OFFSET operations.
 * 
 * Limits the number of rows returned, optionally with an offset.
 */
class PhysicalLimit : public PhysicalOperator {
public:
    /**
     * @brief Construct a new Physical Limit operator.
     * @param op The DuckDB logical limit operator.
     * @param child The child operator to limit.
     */
    PhysicalLimit(duckdb::LogicalLimit& op, std::unique_ptr<PhysicalOperator> child);

    /**
     * @brief Build the streaming node for this operator.
     */
    rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

private:
    int64_t limit_count_;
    int64_t offset_count_;
};

}  // namespace rapidsmpf_duckdb

