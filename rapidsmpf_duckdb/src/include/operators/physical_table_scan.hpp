/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../physical_operator.hpp"

#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

namespace rapidsmpf_duckdb {

/**
 * @brief Physical operator for scanning tables via rapidsmpf.
 * 
 * Converts from DuckDB's LogicalGet operator. Currently supports
 * reading from parquet files through rapidsmpf::streaming::node::read_parquet.
 */
class PhysicalTableScan : public PhysicalOperator {
  public:
    /**
     * @brief Construct a table scan operator.
     * 
     * @param get The LogicalGet operator from DuckDB.
     */
    explicit PhysicalTableScan(duckdb::LogicalGet& get);

    [[nodiscard]] std::string GetName() const override {
        return "RAPIDSMPF_TABLE_SCAN";
    }

    [[nodiscard]] bool IsSource() const noexcept override {
        return true;
    }

    /**
     * @brief Build the streaming node for parquet reading.
     * 
     * Uses rapidsmpf::streaming::node::read_parquet to read data
     * and send to the output channel.
     */
    [[nodiscard]] rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

    /**
     * @brief Get the table name being scanned.
     */
    [[nodiscard]] std::string const& GetTableName() const noexcept {
        return table_name_;
    }

    /**
     * @brief Get the file paths for parquet reading.
     */
    [[nodiscard]] std::vector<std::string> const& GetFilePaths() const noexcept {
        return file_paths_;
    }

    /**
     * @brief Get the column names to read.
     */
    [[nodiscard]] std::vector<std::string> const& GetColumnNames() const noexcept {
        return column_names_;
    }

  private:
    std::string table_name_;
    std::vector<std::string> file_paths_;
    std::vector<std::string> column_names_;
};

}  // namespace rapidsmpf_duckdb



