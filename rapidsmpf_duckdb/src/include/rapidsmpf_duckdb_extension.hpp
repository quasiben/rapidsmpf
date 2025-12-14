/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "duckdb.hpp"

namespace duckdb {

/**
 * @brief DuckDB extension for executing queries via rapidsmpf streaming.
 * 
 * This extension provides a table function `rapidsmpf_query(sql)` that:
 * 1. Parses the SQL query using DuckDB
 * 2. Extracts the LogicalOperator tree
 * 3. Converts it to rapidsmpf streaming operators
 * 4. Executes via rapidsmpf's GPU-accelerated streaming framework
 * 
 * Currently supported operations:
 * - SELECT (projection)
 * - WHERE (filter)
 * - AVG/MEAN aggregation (ungrouped)
 * 
 * TODO:
 * - GROUP BY aggregations
 * - JOIN operations
 * - ORDER BY
 * - LIMIT
 */
class RapidsmpfDuckdbExtension : public Extension {
  public:
    void Load(DuckDB &db) override;
    std::string Name() override;
};

}  // namespace duckdb

