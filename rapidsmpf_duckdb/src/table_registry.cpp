/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/table_registry.hpp"

#include <algorithm>
#include <stdexcept>

namespace rapidsmpf_duckdb {

TableRegistry& TableRegistry::Instance() {
    static TableRegistry instance;
    return instance;
}

std::vector<std::string> TableRegistry::ListParquetFiles(
    std::string const& root_path
) {
    // This follows the pattern from cpp/benchmarks/streaming/ndsh/utils.cpp
    auto root_entry = std::filesystem::directory_entry(
        std::filesystem::path(root_path)
    );
    
    if (!root_entry.exists()) {
        throw std::runtime_error(
            "Path does not exist: " + root_path
        );
    }
    
    if (!root_entry.is_regular_file() && !root_entry.is_directory()) {
        throw std::runtime_error(
            "Path is neither a regular file nor a directory: " + root_path
        );
    }
    
    if (root_entry.is_regular_file()) {
        if (!root_path.ends_with(".parquet")) {
            throw std::runtime_error(
                "File must have .parquet extension: " + root_path
            );
        }
        return {root_path};
    }
    
    // It's a directory - find all parquet files
    std::vector<std::string> result;
    for (auto const& entry : std::filesystem::directory_iterator(root_path)) {
        if (entry.is_regular_file()) {
            auto path = entry.path().string();
            if (path.ends_with(".parquet")) {
                result.push_back(path);
            }
        }
    }
    
    // Sort for deterministic ordering
    std::sort(result.begin(), result.end());
    
    if (result.empty()) {
        throw std::runtime_error(
            "No parquet files found in directory: " + root_path
        );
    }
    
    return result;
}

void TableRegistry::RegisterTable(
    std::string const& table_name,
    std::string const& path,
    std::vector<std::string> column_names
) {
    auto files = ListParquetFiles(path);
    RegisterTableWithFiles(table_name, std::move(files), std::move(column_names));
}

void TableRegistry::RegisterTableWithFiles(
    std::string const& table_name,
    std::vector<std::string> file_paths,
    std::vector<std::string> column_names
) {
    if (table_name.empty()) {
        throw std::runtime_error("Table name cannot be empty");
    }
    
    if (file_paths.empty()) {
        throw std::runtime_error("File paths cannot be empty for table: " + table_name);
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    TableInfo info;
    info.table_name = table_name;
    info.file_paths = std::move(file_paths);
    info.column_names = std::move(column_names);
    
    tables_[table_name] = std::move(info);
}

bool TableRegistry::UnregisterTable(std::string const& table_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return tables_.erase(table_name) > 0;
}

void TableRegistry::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    tables_.clear();
}

std::optional<TableInfo> TableRegistry::GetTable(
    std::string const& table_name
) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tables_.find(table_name);
    if (it != tables_.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool TableRegistry::HasTable(std::string const& table_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tables_.find(table_name) != tables_.end();
}

std::vector<std::string> TableRegistry::GetTableNames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(tables_.size());
    for (auto const& [name, info] : tables_) {
        names.push_back(name);
    }
    return names;
}

}  // namespace rapidsmpf_duckdb

