/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>
#include <cudf/types.hpp>

#include "../physical_operator.hpp"
#include "duckdb/planner/operator/logical_order.hpp"

namespace rapidsmpf_duckdb {

/**
 * @brief Physical operator for ORDER BY operations.
 * 
 * Sorts rows based on specified columns and orders.
 * Note: This is a blocking operator - it must collect all data before sorting.
 */
class PhysicalOrder : public PhysicalOperator {
public:
    /**
     * @brief Construct a new Physical Order operator.
     * @param op The DuckDB logical order operator.
     * @param child The child operator to sort.
     */
    PhysicalOrder(duckdb::LogicalOrder& op, std::unique_ptr<PhysicalOperator> child);

    /**
     * @brief Build the streaming node for this operator.
     */
    rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

private:
    std::vector<cudf::size_type> sort_columns_;
    std::vector<cudf::order> column_orders_;
    std::vector<cudf::null_order> null_orders_;
};

}  // namespace rapidsmpf_duckdb

