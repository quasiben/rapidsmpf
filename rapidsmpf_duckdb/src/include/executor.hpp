/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

#include "duckdb/common/types/data_chunk.hpp"

#include "physical_operator.hpp"

#include <rmm/cuda_stream_pool.hpp>

#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf_duckdb {

/**
 * @brief Executes a rapidsmpf physical plan.
 * 
 * This class takes a tree of PhysicalOperator nodes and executes them
 * via rapidsmpf's streaming framework. Results are converted back to
 * DuckDB DataChunks for return to the caller.
 */
class Executor {
  public:
    /**
     * @brief Construct an executor.
     * 
     * Creates the rapidsmpf streaming context with default settings.
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
     * @return Vector of DuckDB DataChunks containing results.
     */
    [[nodiscard]] std::vector<duckdb::DataChunk> Execute(
        std::unique_ptr<PhysicalOperator> plan
    );

  private:
    std::unique_ptr<rapidsmpf::RmmResourceAdaptor> mr_adaptor_;
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool_;
    std::shared_ptr<rapidsmpf::Statistics> statistics_;
    std::shared_ptr<rapidsmpf::streaming::Context> ctx_;
    std::shared_ptr<rapidsmpf::Communicator> comm_;
    std::shared_ptr<rapidsmpf::BufferResource> br_;
};

}  // namespace rapidsmpf_duckdb


