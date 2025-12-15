/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cudf/types.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf_substrait {

/**
 * @brief Type of physical operator.
 */
enum class PhysicalOperatorType {
    READ,       ///< Read data from a source (table scan)
    FILTER,     ///< Filter rows based on a predicate
    PROJECT,    ///< Project/compute expressions
    AGGREGATE,  ///< Aggregate data (with optional grouping)
    SORT,       ///< Sort data
    FETCH,      ///< Limit/offset (LIMIT clause)
    JOIN,       ///< Join two relations
    SET,        ///< Set operations (UNION, INTERSECT, EXCEPT)
};

/**
 * @brief Convert operator type to string for debugging.
 */
std::string PhysicalOperatorTypeToString(PhysicalOperatorType type);

/**
 * @brief Base class for physical operators in rapidsmpf_substrait.
 *
 * Physical operators form a tree representing the execution plan.
 * Each operator can build a streaming Node for execution in the
 * rapidsmpf streaming framework.
 */
class PhysicalOperator {
  public:
    /**
     * @brief Construct a physical operator.
     *
     * @param type The type of this operator.
     * @param output_types The data types of the output columns.
     * @param estimated_cardinality Estimated number of output rows.
     */
    PhysicalOperator(
        PhysicalOperatorType type,
        std::vector<cudf::data_type> output_types,
        cudf::size_type estimated_cardinality = 0
    );

    virtual ~PhysicalOperator() = default;

    /**
     * @brief Build a streaming node for this operator.
     *
     * Creates a coroutine that processes data according to this operator's
     * semantics. The node reads from ch_in and writes to ch_out.
     *
     * @param ctx The streaming context.
     * @param ch_in Input channel (may be nullptr for source operators).
     * @param ch_out Output channel.
     * @return A streaming node (coroutine).
     */
    virtual rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) = 0;

    /**
     * @brief Check if this operator is a source (has no input).
     */
    [[nodiscard]] virtual bool IsSource() const { return children_.empty(); }

    /**
     * @brief Get the operator type.
     */
    [[nodiscard]] PhysicalOperatorType Type() const { return type_; }

    /**
     * @brief Get the output column types.
     */
    [[nodiscard]] std::vector<cudf::data_type> const& OutputTypes() const { return output_types_; }

    /**
     * @brief Get the estimated cardinality.
     */
    [[nodiscard]] cudf::size_type EstimatedCardinality() const { return estimated_cardinality_; }

    /**
     * @brief Get the child operators.
     */
    [[nodiscard]] std::vector<std::unique_ptr<PhysicalOperator>>& Children() { return children_; }

    /**
     * @brief Get a string description of this operator.
     */
    [[nodiscard]] virtual std::string ToString() const;

  protected:
    PhysicalOperatorType type_;
    std::vector<cudf::data_type> output_types_;
    cudf::size_type estimated_cardinality_;
    std::vector<std::unique_ptr<PhysicalOperator>> children_;
};

}  // namespace rapidsmpf_substrait

