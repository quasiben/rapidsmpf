/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rapidsmpf_duckdb {

/**
 * @brief Information about a registered table.
 */
struct TableInfo {
    std::string table_name;                  ///< Name of the table
    std::vector<std::string> file_paths;     ///< Parquet file paths
    std::vector<std::string> column_names;   ///< Optional column projection
};

/**
 * @brief Registry for mapping DuckDB table names to parquet file paths.
 *
 * This class provides a mechanism to register parquet files or directories
 * as tables that can be referenced by name in SQL queries executed through
 * the rapidsmpf_query function.
 *
 * Usage pattern (following NDSH benchmarks):
 *   1. Register tables with parquet file paths:
 *      registry.RegisterTable("lineitem", "/path/to/tpch/lineitem/");
 *      registry.RegisterTable("nation", "/path/to/tpch/nation/");
 *
 *   2. Run queries referencing these tables:
 *      SELECT * FROM rapidsmpf_query('SELECT * FROM lineitem WHERE ...')
 *
 * The registry automatically discovers parquet files in directories.
 */
class TableRegistry {
  public:
    /**
     * @brief Get the global singleton instance.
     */
    static TableRegistry& Instance();

    /**
     * @brief Register a table from a path.
     *
     * @param table_name Name to use for the table in SQL queries.
     * @param path Path to a parquet file or directory containing parquet files.
     * @param column_names Optional list of columns to project (empty = all columns).
     *
     * @throws std::runtime_error if the path doesn't exist or contains no parquet files.
     */
    void RegisterTable(
        std::string const& table_name,
        std::string const& path,
        std::vector<std::string> column_names = {}
    );

    /**
     * @brief Register a table from explicit file paths.
     *
     * @param table_name Name to use for the table in SQL queries.
     * @param file_paths List of parquet file paths.
     * @param column_names Optional list of columns to project (empty = all columns).
     */
    void RegisterTableWithFiles(
        std::string const& table_name,
        std::vector<std::string> file_paths,
        std::vector<std::string> column_names = {}
    );

    /**
     * @brief Unregister a table.
     *
     * @param table_name Name of the table to unregister.
     * @return true if the table was found and removed, false otherwise.
     */
    bool UnregisterTable(std::string const& table_name);

    /**
     * @brief Clear all registered tables.
     */
    void Clear();

    /**
     * @brief Get information about a registered table.
     *
     * @param table_name Name of the table to look up.
     * @return Table information if found, std::nullopt otherwise.
     */
    [[nodiscard]] std::optional<TableInfo> GetTable(
        std::string const& table_name
    ) const;

    /**
     * @brief Check if a table is registered.
     *
     * @param table_name Name of the table.
     * @return true if the table is registered.
     */
    [[nodiscard]] bool HasTable(std::string const& table_name) const;

    /**
     * @brief Get all registered table names.
     *
     * @return Vector of table names.
     */
    [[nodiscard]] std::vector<std::string> GetTableNames() const;

    /**
     * @brief List all parquet files in a directory.
     *
     * Follows the pattern from cpp/benchmarks/streaming/ndsh/utils.cpp.
     *
     * @param root_path Path to file or directory.
     * @return List of parquet file paths.
     */
    [[nodiscard]] static std::vector<std::string> ListParquetFiles(
        std::string const& root_path
    );

  private:
    TableRegistry() = default;
    ~TableRegistry() = default;
    TableRegistry(TableRegistry const&) = delete;
    TableRegistry& operator=(TableRegistry const&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, TableInfo> tables_;
};

}  // namespace rapidsmpf_duckdb

