/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_table_scan.hpp"
#include "../include/table_registry.hpp"

// Undefine DEBUG to avoid conflict with DuckDB's -DDEBUG flag and rapidsmpf's LOG_LEVEL::DEBUG
#ifdef DEBUG
#undef DEBUG
#endif

#include <cudf/io/parquet.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/main/client_context.hpp"

namespace rapidsmpf_duckdb {

PhysicalTableScan::PhysicalTableScan(duckdb::LogicalGet& get)
    : PhysicalOperator(
          duckdb::PhysicalOperatorType::TABLE_SCAN,
          get.types,
          get.estimated_cardinality
      ) {
    // Try to get the table name from the TableCatalogEntry first
    auto table_entry = get.GetTable();
    if (table_entry) {
        table_name_ = table_entry->name;
    } else {
        // Fallback to the function name
        table_name_ = get.GetName();
    }
    
    // Get column names from the LogicalGet - these correspond to projected columns
    for (auto const& name : get.names) {
        column_names_.push_back(name);
    }
    
    // Look up file paths from the table registry
    auto table_info = TableRegistry::Instance().GetTable(table_name_);
    if (table_info) {
        file_paths_ = table_info->file_paths;
        // If the registry has column projection, use it (unless we already have columns)
        if (column_names_.empty() && !table_info->column_names.empty()) {
            column_names_ = table_info->column_names;
        }
    }
    // If not found in registry, file_paths_ remains empty.
    // BuildNode will throw an informative error when executed.
}

rapidsmpf::streaming::Node PhysicalTableScan::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> /* ch_in */,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    // Use rapidsmpf's read_parquet node
    // This follows the pattern from cpp/benchmarks/streaming/ndsh/q01.cpp
    
    if (file_paths_.empty()) {
        throw std::runtime_error(
            "No file paths configured for table scan. "
            "Use rapidsmpf_register_table() to register parquet files."
        );
    }
    
    // Build parquet reader options
    auto source = cudf::io::source_info(file_paths_);
    auto options = cudf::io::parquet_reader_options::builder(source)
        .columns(column_names_)
        .build();
    
    // Create the parquet reader node
    // The node will read from the parquet files and send TableChunks to ch_out
    return rapidsmpf::streaming::node::read_parquet(
        ctx,
        ch_out,
        1,  // num_producers
        options,
        1024 * 1024,  // 1M rows per chunk
        nullptr  // No filter pushdown for now
    );
}

}  // namespace rapidsmpf_duckdb



