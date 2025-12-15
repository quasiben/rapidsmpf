/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include "physical_operator.hpp"

namespace rapidsmpf_substrait {

/**
 * @brief Result of executing a Substrait plan.
 *
 * Contains the result table and execution statistics.
 */
struct ExecutionResult {
    /// The result table (may be empty for DML operations).
    std::unique_ptr<cudf::table> table;

    /// Column names for the result.
    std::vector<std::string> column_names;

    /// Total number of rows processed.
    size_t rows_processed = 0;

    /// Execution time in milliseconds.
    double execution_time_ms = 0.0;
};

/**
 * @brief Executes a rapidsmpf_substrait physical plan.
 *
 * This class takes a tree of PhysicalOperator nodes and executes them
 * via rapidsmpf's streaming framework. Results are returned as cudf tables.
 */
class Executor {
  public:
    /**
     * @brief Construct an executor with default settings.
     *
     * Creates the rapidsmpf streaming context with default configuration.
     */
    Executor();

    /**
     * @brief Construct an executor with existing resources.
     *
     * @param comm Communicator for distributed execution.
     * @param br Buffer resource for memory management.
     */
    Executor(
        std::shared_ptr<rapidsmpf::Communicator> comm,
        std::shared_ptr<rapidsmpf::BufferResource> br
    );

    ~Executor();

    /**
     * @brief Execute a physical plan.
     *
     * Builds the streaming pipeline from the operator tree and runs it.
     *
     * @param plan Root of the physical operator tree.
     * @return Execution result containing the output table.
     */
    [[nodiscard]] ExecutionResult Execute(std::unique_ptr<PhysicalOperator> plan);

    /**
     * @brief Execute a physical plan and return the result table.
     *
     * Convenience method that returns just the result table.
     *
     * @param plan Root of the physical operator tree.
     * @return The result table.
     */
    [[nodiscard]] std::unique_ptr<cudf::table> ExecuteAndGetTable(
        std::unique_ptr<PhysicalOperator> plan
    );

    /**
     * @brief Get the streaming context.
     */
    [[nodiscard]] std::shared_ptr<rapidsmpf::streaming::Context> GetContext() const {
        return ctx_;
    }

  private:
    std::unique_ptr<rapidsmpf::RmmResourceAdaptor> mr_adaptor_;
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool_;
    std::shared_ptr<rapidsmpf::Statistics> statistics_;
    std::shared_ptr<rapidsmpf::streaming::Context> ctx_;
    std::shared_ptr<rapidsmpf::Communicator> comm_;
    std::shared_ptr<rapidsmpf::BufferResource> br_;
};

}  // namespace rapidsmpf_substrait

