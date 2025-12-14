/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Test: Verify rapidsmpf_query filter execution
 * 
 * This creates a small parquet file and tests the full pipeline:
 *   rapidsmpf_query -> filter -> aggregation
 */

#include <iostream>
#include <chrono>
#include <string>
#include <filesystem>

#include <cuda_runtime.h>

#include "duckdb.hpp"
#include "rapidsmpf_duckdb_extension.hpp"

namespace fs = std::filesystem;

int main() {
    std::cout << "\n=== rapidsmpf_query Filter Test ===" << std::endl;
    
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
    
    // Create temp directory for test parquet file
    std::string temp_dir = "/tmp/rapidsmpf_test_" + std::to_string(getpid());
    fs::create_directories(temp_dir);
    std::string parquet_file = temp_dir + "/test_data.parquet";
    
    std::cout << "Test parquet file: " << parquet_file << std::endl;
    
    // Create DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    
    // Create test data and save to parquet
    std::cout << "\n1. Creating test data (10000 rows)..." << std::endl;
    con.Query(R"(
        CREATE TABLE test_data AS
        SELECT 
            i AS id,
            (random() * 50)::INT AS quantity,
            (random() * 1000)::DECIMAL(10,2) AS price
        FROM generate_series(1, 10000) t(i)
    )");
    con.Query("COPY test_data TO '" + parquet_file + "' (FORMAT PARQUET)");
    std::cout << "   Parquet file created." << std::endl;
    
    // Verify CPU query works first
    std::cout << "\n2. CPU verification (DuckDB native)..." << std::endl;
    auto cpu_result = con.Query(
        "SELECT AVG(price) FROM read_parquet('" + parquet_file + "') WHERE quantity > 25"
    );
    if (cpu_result->HasError()) {
        std::cerr << "CPU Error: " << cpu_result->GetError() << std::endl;
        return 1;
    }
    cpu_result->Print();
    std::cout << "   CPU query OK" << std::endl;
    
    // Test GPU query
    std::cout << "\n3. GPU test (rapidsmpf_query)..." << std::endl;
    
    // Register table with rapidsmpf
    auto reg_result = con.Query(
        "SELECT rapidsmpf_register_table('test_data', '" + parquet_file + "')"
    );
    if (reg_result->HasError()) {
        std::cerr << "Registration Error: " << reg_result->GetError() << std::endl;
        return 1;
    }
    std::cout << "   Table registered." << std::endl;
    
    // Create schema table for type resolution
    con.Query("DROP TABLE test_data");
    con.Query("CREATE TABLE test_data AS SELECT * FROM read_parquet('" + parquet_file + "') LIMIT 0");
    std::cout << "   Schema created." << std::endl;
    
    // Run GPU query with filter
    std::cout << "\n4. Running rapidsmpf_query with filter..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto gpu_result = con.Query(
        "SELECT * FROM rapidsmpf_query('SELECT AVG(price) FROM test_data WHERE quantity > 25')"
    );
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (gpu_result->HasError()) {
        std::cerr << "GPU Error: " << gpu_result->GetError() << std::endl;
        
        // Cleanup
        fs::remove_all(temp_dir);
        return 1;
    }
    
    std::cout << "   GPU Result:" << std::endl;
    gpu_result->Print();
    std::cout << "   Execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "   Rows returned: " << gpu_result->RowCount() << std::endl;
    
    // Cleanup
    con.Query("SELECT rapidsmpf_clear_tables()");
    fs::remove_all(temp_dir);
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    
    return 0;
}

