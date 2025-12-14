/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Example: Registering Parquet Tables for GPU Execution via rapidsmpf
 *
 * This example demonstrates the table registration mechanism that connects
 * DuckDB table names to parquet file paths for rapidsmpf to read.
 *
 * The workflow is:
 *   1. Load the rapidsmpf_duckdb extension
 *   2. Register parquet files/directories as tables using rapidsmpf_register_table()
 *   3. Execute SQL queries via rapidsmpf_query() referencing those tables
 *
 * This follows the pattern from cpp/benchmarks/streaming/ndsh/ where parquet
 * paths are resolved to streaming operators.
 *
 * Build:
 *   cd /datasets/bzaitlen/GitRepos/rapidsmpf-substrait/rapidsmpf_duckdb
 *   conda activate 2025-12-12-rapidsmpf-duckdb
 *   make release  # Build the extension
 *
 * Run:
 *   ./examples/example_registered_tables
 */

#include <iostream>
#include <chrono>
#include <string>

#include <cuda_runtime.h>

#include "duckdb.hpp"
#include "rapidsmpf_duckdb_extension.hpp"

using namespace duckdb;

void print_separator() {
    std::cout << std::string(70, '=') << std::endl;
}

void print_gpu_memory(const std::string& label) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    double used_mb = (total_bytes - free_bytes) / (1024.0 * 1024.0);
    double total_mb = total_bytes / (1024.0 * 1024.0);
    std::cout << "[GPU Memory] " << label << ": " 
              << used_mb << " MB used / " << total_mb << " MB total" << std::endl;
}

void run_query(Connection& con, const std::string& description, const std::string& sql) {
    print_separator();
    std::cout << description << std::endl;
    std::cout << "SQL: " << sql << std::endl;
    print_separator();
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = con.Query(sql);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (result->HasError()) {
        std::cerr << "Error: " << result->GetError() << std::endl;
        return;
    }
    
    std::cout << "Result (" << result->RowCount() << " rows):" << std::endl;
    result->Print();
    std::cout << "Execution time: " << duration_ms << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default to TPCH scale-1000 data
    std::string data_dir = "/raid/rapidsmpf/data/tpch/scale-1000";
    if (argc > 1) {
        data_dir = argv[1];
    }
    
    std::cout << "\n";
    print_separator();
    std::cout << "RapidsMPF DuckDB Extension - Table Registration Example" << std::endl;
    print_separator();
    std::cout << "\nData directory: " << data_dir << std::endl;
    
    // Initialize CUDA
    int device_count;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found: " << cudaGetErrorString(cuda_status) << std::endl;
        return 1;
    }
    cudaSetDevice(0);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "CUDA Device: " << props.name << std::endl;
    print_gpu_memory("Initial");
    
    // Create DuckDB database and connection
    std::cout << "\nCreating DuckDB in-memory database..." << std::endl;
    DuckDB db(nullptr);
    Connection con(db);
    
    // The extension is already linked into DuckDB, so we don't need to load it dynamically.
    // Just verify it's available by checking if the function exists.
    std::cout << "Checking rapidsmpf_duckdb extension availability..." << std::endl;
    try {
        // Try to list tables - if the function exists, extension is loaded
        auto check = con.Query("SELECT * FROM rapidsmpf_list_tables()");
        if (check->HasError()) {
            // Function not found - try to load extension
            db.LoadExtension<RapidsmpfDuckdbExtension>();
        }
        std::cout << "Extension is available." << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Warning: " << e.what() << std::endl;
        std::cout << "Extension may already be loaded - continuing..." << std::endl;
    }
    
    print_gpu_memory("After extension load");
    
    // =========================================================================
    // Step 1: Register tables from parquet directories
    // =========================================================================
    std::cout << "\n";
    print_separator();
    std::cout << "Registering TPCH Tables" << std::endl;
    print_separator();
    
    // Register each table - maps table name to parquet directory
    // Also create DuckDB views so the planner can resolve table names
    std::vector<std::pair<std::string, std::string>> tables = {
        {"nation", data_dir + "/nation"},
        {"region", data_dir + "/region"},
        {"customer", data_dir + "/customer"},
        {"orders", data_dir + "/orders"},
        {"lineitem", data_dir + "/lineitem"},
        {"part", data_dir + "/part"},
        {"partsupp", data_dir + "/partsupp"},
        {"supplier", data_dir + "/supplier"}
    };
    
    for (auto const& [name, path] : tables) {
        try {
            // Register with our registry for rapidsmpf file paths
            auto result = con.Query(
                "SELECT rapidsmpf_register_table('" + name + "', '" + path + "')"
            );
            if (result->HasError()) {
                std::cerr << "Failed to register " << name << ": " << result->GetError() << std::endl;
                continue;
            }
            
            // Create an actual table in DuckDB using the parquet schema
            // This is done by first reading the parquet metadata to get the schema,
            // then creating the table from a LIMIT 0 query (no data, just schema)
            auto table_result = con.Query(
                "CREATE OR REPLACE TABLE " + name + " AS SELECT * FROM read_parquet('" + path + "/*.parquet') LIMIT 0"
            );
            if (table_result->HasError()) {
                std::cerr << "Failed to create table for " << name << ": " << table_result->GetError() << std::endl;
            } else {
                std::cout << "  Registered: " << name << " -> " << path << std::endl;
            }
        } catch (std::exception& e) {
            std::cerr << "Exception registering " << name << ": " << e.what() << std::endl;
        }
    }
    
    // =========================================================================
    // Step 2: List registered tables
    // =========================================================================
    run_query(con, "List Registered Tables", "SELECT * FROM rapidsmpf_list_tables()");
    
    // =========================================================================
    // Step 3: First verify data with standard DuckDB (for comparison)
    // =========================================================================
    std::cout << "\n";
    print_separator();
    std::cout << "Verifying Data with Standard DuckDB (CPU)" << std::endl;
    print_separator();
    
    run_query(con, "Nation table (CPU read)",
              "SELECT * FROM read_parquet('" + data_dir + "/nation/*.parquet') LIMIT 5");
    
    run_query(con, "Region table (CPU read)",
              "SELECT * FROM read_parquet('" + data_dir + "/region/*.parquet') LIMIT 5");
    
    // =========================================================================
    // Step 4: Execute queries via rapidsmpf
    // =========================================================================
    std::cout << "\n";
    print_separator();
    std::cout << "Executing Queries via RapidsMPF (GPU)" << std::endl;
    print_separator();
    
    print_gpu_memory("Before rapidsmpf queries");
    
    // Simple scan query
    run_query(con, "Query via rapidsmpf: SELECT from nation",
              "SELECT * FROM rapidsmpf_query('SELECT * FROM nation')");
    
    print_gpu_memory("After nation query");
    
    // Aggregation query
    run_query(con, "Query via rapidsmpf: COUNT from region",
              "SELECT * FROM rapidsmpf_query('SELECT COUNT(*) FROM region')");
    
    print_gpu_memory("After region query");
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    std::cout << "\n";
    run_query(con, "Clear all registered tables", "SELECT rapidsmpf_clear_tables()");
    run_query(con, "Verify tables cleared", "SELECT * FROM rapidsmpf_list_tables()");
    
    print_separator();
    std::cout << "Example completed!" << std::endl;
    print_separator();
    
    return 0;
}

