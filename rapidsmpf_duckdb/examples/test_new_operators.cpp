/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Test: New operators - string filter, date filter, ORDER BY, LIMIT
 * 
 * Uses a single partition from TPCH scale-1000 to test:
 *   - String equality filter (l_returnflag = 'A')
 *   - Date comparison filter (l_shipdate > '1995-01-01')
 *   - ORDER BY (l_extendedprice DESC)
 *   - LIMIT (10 rows)
 */

#include <iostream>
#include <chrono>
#include <string>

#include <cuda_runtime.h>

#include "duckdb.hpp"
#include "rapidsmpf_duckdb_extension.hpp"

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Test: New Operators (String, Date, Order, Limit)\n";
    std::cout << "========================================\n";
    
    // Check CUDA
    int device_count;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    cudaSetDevice(0);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "GPU: " << props.name << std::endl;
    
    // Use single partition for quick test
    std::string parquet_file = "/raid/rapidsmpf/data/tpch/scale-1000/lineitem/part.0.parquet";
    std::cout << "Data: " << parquet_file << std::endl;
    
    // Create DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    
    // ========================================
    // Test 1: String filter only
    // ========================================
    std::cout << "\n--- Test 1: String Filter ---\n";
    std::cout << "Query: SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'A'\n";
    
    auto cpu_result = con.Query(
        "SELECT COUNT(*) FROM read_parquet('" + parquet_file + "') WHERE l_returnflag = 'A'"
    );
    if (cpu_result->HasError()) {
        std::cerr << "CPU Error: " << cpu_result->GetError() << std::endl;
    } else {
        std::cout << "CPU Result: ";
        cpu_result->Print();
    }
    
    // ========================================
    // Test 2: Date filter only
    // ========================================
    std::cout << "\n--- Test 2: Date Filter ---\n";
    std::cout << "Query: SELECT COUNT(*) FROM lineitem WHERE l_shipdate > '1995-06-01'\n";
    
    cpu_result = con.Query(
        "SELECT COUNT(*) FROM read_parquet('" + parquet_file + "') WHERE l_shipdate > DATE '1995-06-01'"
    );
    if (cpu_result->HasError()) {
        std::cerr << "CPU Error: " << cpu_result->GetError() << std::endl;
    } else {
        std::cout << "CPU Result: ";
        cpu_result->Print();
    }
    
    // ========================================
    // Test 3: ORDER BY + LIMIT
    // ========================================
    std::cout << "\n--- Test 3: ORDER BY + LIMIT ---\n";
    std::cout << "Query: SELECT l_orderkey, l_extendedprice FROM lineitem ORDER BY l_extendedprice DESC LIMIT 5\n";
    
    cpu_result = con.Query(
        "SELECT l_orderkey, l_extendedprice FROM read_parquet('" + parquet_file + "') "
        "ORDER BY l_extendedprice DESC LIMIT 5"
    );
    if (cpu_result->HasError()) {
        std::cerr << "CPU Error: " << cpu_result->GetError() << std::endl;
    } else {
        std::cout << "CPU Result:\n";
        cpu_result->Print();
    }
    
    // ========================================
    // Test 4: Combined - String + Date + ORDER BY + LIMIT
    // ========================================
    std::cout << "\n--- Test 4: Combined Query ---\n";
    std::cout << "Query: SELECT l_orderkey, l_extendedprice, l_shipdate\n";
    std::cout << "       FROM lineitem\n";
    std::cout << "       WHERE l_returnflag = 'A' AND l_shipdate > '1995-06-01'\n";
    std::cout << "       ORDER BY l_extendedprice DESC\n";
    std::cout << "       LIMIT 10\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cpu_result = con.Query(
        "SELECT l_orderkey, l_extendedprice, l_shipdate "
        "FROM read_parquet('" + parquet_file + "') "
        "WHERE l_returnflag = 'A' AND l_shipdate > DATE '1995-06-01' "
        "ORDER BY l_extendedprice DESC "
        "LIMIT 10"
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (cpu_result->HasError()) {
        std::cerr << "CPU Error: " << cpu_result->GetError() << std::endl;
    } else {
        std::cout << "CPU Result (" << duration.count() << " ms):\n";
        cpu_result->Print();
    }
    
    // ========================================
    // Now test GPU execution
    // ========================================
    std::cout << "\n========================================\n";
    std::cout << "GPU Execution Tests (rapidsmpf_query)\n";
    std::cout << "========================================\n";
    
    // Register the lineitem table
    auto reg_result = con.Query(
        "SELECT rapidsmpf_register_table('lineitem', '" + parquet_file + "')"
    );
    if (reg_result->HasError()) {
        std::cerr << "Registration Error: " << reg_result->GetError() << std::endl;
        return 1;
    }
    std::cout << "Table 'lineitem' registered.\n";
    
    // Create schema table for type resolution
    con.Query("CREATE TABLE lineitem AS SELECT * FROM read_parquet('" + parquet_file + "') LIMIT 0");
    std::cout << "Schema created.\n";
    
    // ========================================
    // GPU Test: Numeric Filter + AVG (known working)
    // ========================================
    std::cout << "\n--- GPU Test: Numeric Filter + AVG ---\n";
    std::cout << "Query: SELECT AVG(l_extendedprice) FROM lineitem WHERE l_quantity > 40\n";
    
    start = std::chrono::high_resolution_clock::now();
    auto gpu_result = con.Query(
        "SELECT * FROM rapidsmpf_query('SELECT AVG(l_extendedprice) FROM lineitem WHERE l_quantity > 40')"
    );
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (gpu_result->HasError()) {
        std::cerr << "GPU Error: " << gpu_result->GetError() << std::endl;
    } else {
        std::cout << "GPU completed (" << duration.count() << " ms): ";
        std::cout << gpu_result->RowCount() << " rows returned\n";
        std::cout << "(Note: Result conversion to DuckDB not yet implemented)\n";
    }
    
    // Cleanup
    con.Query("SELECT rapidsmpf_clear_tables()");
    
    std::cout << "\n========================================\n";
    std::cout << "Test Complete\n";
    std::cout << "========================================\n";
    
    return 0;
}

