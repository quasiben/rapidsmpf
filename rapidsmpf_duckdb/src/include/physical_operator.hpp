/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "duckdb/common/types.hpp"
#include "duckdb/common/enums/physical_operator_type.hpp"

// Undefine DEBUG to avoid conflict with DuckDB's -DDEBUG flag and rapidsmpf's LOG_LEVEL::DEBUG
#ifdef DEBUG
#define RAPIDSMPF_DUCKDB_DEBUG_WAS_DEFINED
#undef DEBUG
#endif

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf_duckdb {

/**
 * @brief Base class for physical operators that execute via rapidsmpf streaming.
 * 
 * Each physical operator represents a stage in the streaming execution pipeline.
 * The operator creates a streaming Node (coroutine) that reads from input
 * channels and writes to output channels.
 */
class PhysicalOperator {
  public:
    PhysicalOperator(
        duckdb::PhysicalOperatorType type,
        std::vector<duckdb::LogicalType> types,
        duckdb::idx_t estimated_cardinality
    );

    virtual ~PhysicalOperator() = default;

    // Non-copyable
    PhysicalOperator(PhysicalOperator const&) = delete;
    PhysicalOperator& operator=(PhysicalOperator const&) = delete;
    PhysicalOperator(PhysicalOperator&&) = default;
    PhysicalOperator& operator=(PhysicalOperator&&) = default;

    /**
     * @brief Get the operator type.
     */
    [[nodiscard]] duckdb::PhysicalOperatorType GetType() const noexcept {
        return type_;
    }

    /**
     * @brief Get the output types.
     */
    [[nodiscard]] std::vector<duckdb::LogicalType> const& GetTypes() const noexcept {
        return types_;
    }

    /**
     * @brief Get the estimated cardinality.
     */
    [[nodiscard]] duckdb::idx_t GetEstimatedCardinality() const noexcept {
        return estimated_cardinality_;
    }

    /**
     * @brief Get the human-readable name.
     */
    [[nodiscard]] virtual std::string GetName() const;

    /**
     * @brief Check if this is a source operator.
     */
    [[nodiscard]] virtual bool IsSource() const noexcept {
        return false;
    }

    /**
     * @brief Check if this is a sink operator.
     */
    [[nodiscard]] virtual bool IsSink() const noexcept {
        return false;
    }

    /**
     * @brief Get child operators.
     */
    [[nodiscard]] std::vector<std::unique_ptr<PhysicalOperator>>& Children() noexcept {
        return children_;
    }

    /**
     * @brief Add a child operator.
     */
    void AddChild(std::unique_ptr<PhysicalOperator> child);

    /**
     * @brief Build the streaming node for this operator.
     * 
     * @param ctx Streaming context.
     * @param ch_in Input channel (nullptr for sources).
     * @param ch_out Output channel (nullptr for sinks).
     * @return Streaming node coroutine.
     */
    [[nodiscard]] virtual rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) = 0;

    /**
     * @brief Convert to string for debugging.
     */
    [[nodiscard]] virtual std::string ToString(int indent = 0) const;

  protected:
    duckdb::PhysicalOperatorType type_;
    std::vector<duckdb::LogicalType> types_;
    duckdb::idx_t estimated_cardinality_;
    std::vector<std::unique_ptr<PhysicalOperator>> children_;
};

}  // namespace rapidsmpf_duckdb



