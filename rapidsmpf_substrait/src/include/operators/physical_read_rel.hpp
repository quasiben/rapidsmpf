/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "../physical_operator.hpp"
#include "../substrait_plan_converter.hpp"

namespace rapidsmpf_substrait {

/**
 * @brief Physical operator for reading data (table scan).
 *
 * Corresponds to Substrait's ReadRel. Supports:
 * - Named tables (looked up in table registry)
 * - Local files (parquet)
 * - Column projection
 * - Filter pushdown
 */
class PhysicalReadRel : public PhysicalOperator {
  public:
    /**
     * @brief Construct a read operator.
     *
     * @param table_name The logical table name (for named tables).
     * @param file_paths Explicit file paths (for local files).
     * @param column_names Names of all columns in the table.
     * @param column_types Types of all columns in the table.
     * @param projected_columns Indices of columns to read (empty = all).
     * @param filter_expr Optional filter expression to push down.
     */
    PhysicalReadRel(
        std::string table_name,
        std::vector<std::string> file_paths,
        std::vector<std::string> column_names,
        std::vector<cudf::data_type> column_types,
        std::vector<int32_t> projected_columns,
        std::optional<ExpressionInfo> filter_expr = std::nullopt
    );

    [[nodiscard]] rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) override;

    [[nodiscard]] bool IsSource() const override { return true; }

    [[nodiscard]] std::string ToString() const override;

    /**
     * @brief Get the table name.
     */
    [[nodiscard]] std::string const& TableName() const { return table_name_; }

    /**
     * @brief Get the file paths.
     */
    [[nodiscard]] std::vector<std::string> const& FilePaths() const { return file_paths_; }

  private:
    std::string table_name_;
    std::vector<std::string> file_paths_;
    std::vector<std::string> column_names_;
    std::vector<int32_t> projected_columns_;
    std::optional<ExpressionInfo> filter_expr_;
};

}  // namespace rapidsmpf_substrait

