/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * GPU Example: Testing rapidsmpf_duckdb extension with GPU monitoring
 * 
 * This example demonstrates:
 * 1. Loading the rapidsmpf_duckdb extension
 * 2. Using rapidsmpf_query to execute queries on GPU
 * 3. Monitoring GPU memory usage
 */

#include <iostream>
#include <chrono>
#include <string>

#include "duckdb.hpp"

#include <cuda_runtime.h>

void print_separator() {
    std::cout << std::string(70, '=') << std::endl;
}

// GPU memory monitoring
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
    std::cout << "[GPU Memory] " << label << ": "
              << "Used: " << (mem.used_bytes / (1024.0 * 1024.0)) << " MB, "
              << "Free: " << (mem.free_bytes / (1024.0 * 1024.0)) << " MB"
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
        // Use rapidsmpf_query table function
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
    
    std::cout << "[GPU Memory] Before: " << (mem_before.used_bytes / (1024.0 * 1024.0)) 
              << " MB, After: " << (mem_after.used_bytes / (1024.0 * 1024.0)) << " MB"
              << ", Delta: " << ((mem_after.used_bytes - mem_before.used_bytes) / (1024.0 * 1024.0)) << " MB"
              << std::endl;
    
    if (result->HasError()) {
        std::cerr << "Error: " << result->GetError() << std::endl;
        return;
    }
    
    std::cout << "\nResult (" << result->RowCount() << " rows):" << std::endl;
    result->Print();
    std::cout << "\nExecution time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << std::endl;
    print_separator();
    std::cout << "RapidsMPF DuckDB Extension - GPU Example" << std::endl;
    print_separator();
    std::cout << std::endl;
    
    // Check CUDA device
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found! Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total GPU Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    
    print_gpu_memory("Initial state");
    std::cout << std::endl;
    
    // Create in-memory database
    std::cout << "Creating DuckDB in-memory database..." << std::endl;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    
    print_gpu_memory("After DuckDB init");
    
    // Create sample tables
    std::cout << "\nCreating sample tables..." << std::endl;
    
    con.Query(R"(
        CREATE TABLE orders AS
        SELECT
            i AS order_id,
            (i % 1000) + 1 AS customer_id,
            (i % 100) + 1 AS product_id,
            (random() * 9 + 1)::INT AS quantity,
            (random() * 990 + 10)::DECIMAL(10,2) AS price,
            DATE '2024-01-01' + INTERVAL (i % 365) DAY AS order_date
        FROM generate_series(1, 100000) t(i)
    )");
    std::cout << "  - orders: 100,000 rows" << std::endl;
    
    con.Query(R"(
        CREATE TABLE customers AS
        SELECT
            i AS customer_id,
            'Customer_' || i AS name,
            CASE (i % 4)
                WHEN 0 THEN 'North'
                WHEN 1 THEN 'South'
                WHEN 2 THEN 'East'
                ELSE 'West'
            END AS region
        FROM generate_series(1, 1000) t(i)
    )");
    std::cout << "  - customers: 1,000 rows" << std::endl;
    
    std::cout << "\nTables created!" << std::endl;
    print_gpu_memory("After table creation");
    std::cout << std::endl;
    
    // Test 1: Standard DuckDB query (CPU)
    run_query(con, "Test 1: Simple Aggregation (CPU)", 
        "SELECT COUNT(*) as cnt, SUM(price) as total, AVG(price) as avg_price FROM orders",
        false  // CPU mode
    );
    
    // Test 2: Try rapidsmpf_query (GPU)
    std::cout << "\n>>> Attempting to use rapidsmpf_query function...\n" << std::endl;
    run_query(con, "Test 2: Simple Aggregation (rapidsmpf)", 
        "SELECT COUNT(*) as cnt, SUM(price) as total, AVG(price) as avg_price FROM orders",
        true  // GPU mode via rapidsmpf_query
    );
    
    // Test 3: Another CPU query for comparison
    run_query(con, "Test 3: GROUP BY Query (CPU)",
        R"(
        SELECT customer_id, COUNT(*) AS cnt, SUM(price) AS total
        FROM orders
        GROUP BY customer_id
        ORDER BY total DESC
        LIMIT 10
        )",
        false
    );
    
    // GPU Memory allocation test
    print_separator();
    std::cout << "GPU Memory Allocation Test (using cudaMalloc)" << std::endl;
    print_separator();
    
    print_gpu_memory("Before allocation");
    
    // Allocate some GPU memory directly via CUDA
    void* d_ptr = nullptr;
    size_t alloc_size = 100 * 1024 * 1024;  // 100 MB
    
    err = cudaMalloc(&d_ptr, alloc_size);
    if (err == cudaSuccess) {
        std::cout << "Allocated " << (alloc_size / (1024.0 * 1024.0)) << " MB on GPU" << std::endl;
        print_gpu_memory("After allocation");
        
        cudaFree(d_ptr);
        std::cout << "Deallocated memory" << std::endl;
        print_gpu_memory("After deallocation");
    } else {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    std::cout << std::endl;
    print_separator();
    std::cout << "Example completed!" << std::endl;
    print_separator();
    
    return 0;
}
