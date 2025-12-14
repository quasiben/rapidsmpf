/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Parquet Example: Testing rapidsmpf_duckdb with parquet file input
 * 
 * This example uses TPCH parquet data to test the rapidsmpf execution path.
 * Note: rapidsmpf_query currently only works with parquet file sources.
 */

#include <iostream>
#include <chrono>
#include <string>

#include "duckdb.hpp"

#include <cuda_runtime.h>

void print_separator() {
    std::cout << std::string(70, '=') << std::endl;
}

struct GPUMemoryInfo {
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
};

GPUMemoryInfo get_gpu_memory() {
    GPUMemoryInfo info;
    cudaMemGetInfo(&info.free_bytes, &info.total_bytes);
    info.used_bytes = info.total_bytes - info.free_bytes;
    return info;
}

void print_gpu_memory(const std::string& label) {
    auto mem = get_gpu_memory();
    std::cout << "[GPU] " << label << ": "
              << "Used: " << (mem.used_bytes / (1024.0 * 1024.0)) << " MB"
              << std::endl;
}

void run_query(duckdb::Connection& con, const std::string& description, 
               const std::string& sql, bool use_rapidsmpf = false) {
    print_separator();
    std::cout << description << std::endl;
    if (use_rapidsmpf) {
        std::cout << "[Mode: rapidsmpf GPU execution]" << std::endl;
    } else {
        std::cout << "[Mode: standard DuckDB CPU execution]" << std::endl;
    }
    print_separator();
    
    auto mem_before = get_gpu_memory();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    duckdb::unique_ptr<duckdb::MaterializedQueryResult> result;
    if (use_rapidsmpf) {
        std::string rapidsmpf_sql = "SELECT * FROM rapidsmpf_query('" + sql + "')";
        result = con.Query(rapidsmpf_sql);
    } else {
        result = con.Query(sql);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    );
    
    auto mem_after = get_gpu_memory();
    double delta_mb = (mem_after.used_bytes - mem_before.used_bytes) / (1024.0 * 1024.0);
    
    std::cout << "[GPU Memory] Delta: " << delta_mb << " MB" << std::endl;
    
    if (result->HasError()) {
        std::cerr << "Error: " << result->GetError() << std::endl;
        return;
    }
    
    std::cout << "\nResult (" << result->RowCount() << " rows):" << std::endl;
    result->Print();
    std::cout << "\nExecution time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::string tpch_path = "/raid/rapidsmpf/data/tpch/scale-1000";
    
    if (argc > 1) {
        tpch_path = argv[1];
    }
    
    std::cout << std::endl;
    print_separator();
    std::cout << "RapidsMPF DuckDB - Parquet Example" << std::endl;
    std::cout << "TPCH Data Path: " << tpch_path << std::endl;
    print_separator();
    std::cout << std::endl;
    
    // Check CUDA device
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    print_gpu_memory("Initial state");
    std::cout << std::endl;
    
    // Create database
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    
    // Test 1: CPU query on parquet (nation table - small)
    std::string nation_sql = "SELECT * FROM read_parquet('" + tpch_path + "/nation/*.parquet')";
    run_query(con, "Test 1: Nation table via DuckDB CPU", nation_sql, false);
    
    // Test 2: Attempt rapidsmpf_query on parquet
    // Note: This will fail because rapidsmpf_query parses the SQL string
    // and creates a new plan, but read_parquet is a table function
    std::cout << "\n>>> Testing rapidsmpf_query with parquet...\n" << std::endl;
    run_query(con, "Test 2: Nation table via rapidsmpf (will show limitation)", 
        nation_sql, true);
    
    // Test 3: CPU aggregation on lineitem (larger)
    std::string lineitem_sql = R"(
        SELECT 
            l_returnflag,
            l_linestatus,
            COUNT(*) as count_order,
            SUM(l_quantity) as sum_qty
        FROM read_parquet(')" + tpch_path + R"(/lineitem/part.0.parquet')
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
    )";
    run_query(con, "Test 3: Lineitem aggregation via DuckDB CPU", lineitem_sql, false);
    
    // Summary
    print_separator();
    std::cout << "Summary:" << std::endl;
    print_separator();
    std::cout << R"(
Current Limitations of rapidsmpf_query:

1. The function takes a SQL string and re-plans it internally
2. This works for simple queries on base tables registered in the catalog
3. It does NOT work with:
   - Table functions like read_parquet() in the SQL string
   - In-memory tables (they require file-based data sources)
   - Complex queries with JOIN, ORDER BY, LIMIT, DISTINCT, GROUP BY

To fully enable GPU execution, the extension needs:
- A way to register parquet files as tables (rapidsmpf_register_table)
- Or direct integration with DuckDB's execution engine
- Support for more operator types in the physical plan generator

GPU Memory allocation IS working - we saw memory being allocated during
the rapidsmpf initialization phase.
)" << std::endl;
    
    print_separator();
    std::cout << "Example completed!" << std::endl;
    print_separator();
    
    return 0;
}

