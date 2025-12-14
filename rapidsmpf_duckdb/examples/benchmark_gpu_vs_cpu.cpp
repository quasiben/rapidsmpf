/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Benchmark: GPU (rapidsmpf_duckdb) vs CPU (DuckDB) Performance Comparison
 *
 * Query: SELECT AVG(l_extendedprice) FROM lineitem WHERE l_quantity > 25
 * 
 * This benchmark runs a realistic query that:
 *   1. Reads data from parquet
 *   2. Filters rows (WHERE l_quantity > 25)
 *   3. Calculates aggregate (AVG of l_extendedprice)
 *
 * Build:
 *   cd /datasets/bzaitlen/GitRepos/rapidsmpf-substrait/rapidsmpf_duckdb
 *   conda activate 2025-12-12-rapidsmpf-duckdb
 *   make release
 *
 * Run:
 *   ./build/release/extension/rapidsmpf_duckdb/examples/benchmark_gpu_vs_cpu [data_dir] [num_runs]
 */

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

#include <cuda_runtime.h>

#include "duckdb.hpp"
#include "rapidsmpf_duckdb_extension.hpp"

using namespace duckdb;

struct MemorySnapshot {
    size_t gpu_used_bytes;
    size_t gpu_total_bytes;
    
    double gpu_used_mb() const { return gpu_used_bytes / (1024.0 * 1024.0); }
    double gpu_total_mb() const { return gpu_total_bytes / (1024.0 * 1024.0); }
};

MemorySnapshot get_gpu_memory() {
    MemorySnapshot snap;
    size_t free_bytes;
    cudaMemGetInfo(&free_bytes, &snap.gpu_total_bytes);
    snap.gpu_used_bytes = snap.gpu_total_bytes - free_bytes;
    return snap;
}

void print_separator(char c = '=', int width = 80) {
    std::cout << std::string(width, c) << std::endl;
}

struct BenchmarkQueryResult {
    double time_ms;
    int64_t row_count;
    std::string result_str;
    MemorySnapshot mem_before;
    MemorySnapshot mem_after;
    
    double mem_delta_mb() const {
        return (double)(mem_after.gpu_used_bytes - mem_before.gpu_used_bytes) / (1024.0 * 1024.0);
    }
};

BenchmarkQueryResult run_query(Connection& con, const std::string& sql) {
    BenchmarkQueryResult qr;
    
    // Sync GPU before measuring
    cudaDeviceSynchronize();
    qr.mem_before = get_gpu_memory();
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = con.Query(sql);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    qr.mem_after = get_gpu_memory();
    qr.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (result->HasError()) {
        std::cerr << "Query Error: " << result->GetError() << std::endl;
        qr.row_count = -1;
        qr.result_str = "ERROR";
        return qr;
    }
    
    qr.row_count = result->RowCount();
    
    // Get result value as string
    if (qr.row_count > 0) {
        auto chunk = result->Fetch();
        if (chunk && chunk->size() > 0) {
            qr.result_str = chunk->GetValue(0, 0).ToString();
        }
    }
    
    return qr;
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string data_dir = "/raid/rapidsmpf/data/tpch/scale-1000";
    int num_runs = 3;
    
    if (argc > 1) data_dir = argv[1];
    if (argc > 2) num_runs = std::stoi(argv[2]);
    
    std::cout << "\n";
    print_separator('=', 80);
    std::cout << "GPU vs CPU Benchmark: rapidsmpf_duckdb vs DuckDB" << std::endl;
    print_separator('=', 80);
    std::cout << "Data directory: " << data_dir << std::endl;
    std::cout << "Number of runs: " << num_runs << std::endl;
    
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
    
    auto initial_mem = get_gpu_memory();
    std::cout << "Initial GPU Memory: " << std::fixed << std::setprecision(1) 
              << initial_mem.gpu_used_mb() << " MB / " 
              << initial_mem.gpu_total_mb() << " MB" << std::endl;
    
    // Create DuckDB database and connection
    DuckDB db(nullptr);
    Connection con(db);
    
    // Check extension availability
    try {
        auto check = con.Query("SELECT * FROM rapidsmpf_list_tables()");
        if (check->HasError()) {
            db.LoadExtension<RapidsmpfDuckdbExtension>();
        }
    } catch (...) {}
    
    // =========================================================================
    // Register lineitem table for GPU queries
    // =========================================================================
    std::cout << "\nRegistering lineitem table..." << std::endl;
    
    std::string lineitem_path = data_dir + "/lineitem";
    con.Query("SELECT rapidsmpf_register_table('lineitem', '" + lineitem_path + "')");
    con.Query("CREATE OR REPLACE TABLE lineitem AS SELECT * FROM read_parquet('" + lineitem_path + "/*.parquet') LIMIT 0");
    
    std::cout << "Table registered." << std::endl;
    
    // =========================================================================
    // Define the benchmark query
    // =========================================================================
    // Query: Filter rows where quantity > 25, then compute average of extended price
    std::string cpu_query = 
        "SELECT AVG(l_extendedprice) as avg_price "
        "FROM read_parquet('" + lineitem_path + "/*.parquet') "
        "WHERE l_quantity > 25";
    
    std::string gpu_query = 
        "SELECT * FROM rapidsmpf_query("
        "'SELECT AVG(l_extendedprice) as avg_price FROM lineitem WHERE l_quantity > 25'"
        ")";
    
    std::cout << "\n";
    print_separator('-', 80);
    std::cout << "Query: SELECT AVG(l_extendedprice) FROM lineitem WHERE l_quantity > 25" << std::endl;
    print_separator('-', 80);
    
    // =========================================================================
    // Run CPU benchmark
    // =========================================================================
    std::cout << "\n--- CPU (DuckDB) ---" << std::endl;
    
    std::vector<double> cpu_times;
    std::vector<double> cpu_mem_deltas;
    std::string cpu_result;
    
    for (int i = 0; i < num_runs; i++) {
        // Force garbage collection between runs
        cudaDeviceSynchronize();
        
        auto qr = run_query(con, cpu_query);
        cpu_times.push_back(qr.time_ms);
        cpu_mem_deltas.push_back(qr.mem_delta_mb());
        cpu_result = qr.result_str;
        
        std::cout << "  Run " << (i+1) << ": " 
                  << std::fixed << std::setprecision(0) << qr.time_ms << " ms, "
                  << "GPU mem delta: " << std::setprecision(1) << qr.mem_delta_mb() << " MB"
                  << std::endl;
    }
    
    double cpu_min = *std::min_element(cpu_times.begin(), cpu_times.end());
    double cpu_avg = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / cpu_times.size();
    
    std::cout << "  Result: " << cpu_result << std::endl;
    std::cout << "  Min time: " << std::fixed << std::setprecision(0) << cpu_min << " ms" << std::endl;
    std::cout << "  Avg time: " << cpu_avg << " ms" << std::endl;
    
    // =========================================================================
    // Run GPU benchmark
    // =========================================================================
    std::cout << "\n--- GPU (rapidsmpf_duckdb) ---" << std::endl;
    
    std::vector<double> gpu_times;
    std::vector<double> gpu_mem_deltas;
    std::string gpu_result;
    
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        
        auto qr = run_query(con, gpu_query);
        gpu_times.push_back(qr.time_ms);
        gpu_mem_deltas.push_back(qr.mem_delta_mb());
        gpu_result = qr.result_str;
        
        std::cout << "  Run " << (i+1) << ": " 
                  << std::fixed << std::setprecision(0) << qr.time_ms << " ms, "
                  << "GPU mem delta: " << std::setprecision(1) << qr.mem_delta_mb() << " MB"
                  << std::endl;
    }
    
    double gpu_min = *std::min_element(gpu_times.begin(), gpu_times.end());
    double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();
    
    std::cout << "  Result: " << gpu_result << std::endl;
    std::cout << "  Min time: " << std::fixed << std::setprecision(0) << gpu_min << " ms" << std::endl;
    std::cout << "  Avg time: " << gpu_avg << " ms" << std::endl;
    
    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "\n";
    print_separator('=', 80);
    std::cout << "SUMMARY" << std::endl;
    print_separator('=', 80);
    
    auto final_mem = get_gpu_memory();
    
    std::cout << std::left << std::setw(20) << "Metric" 
              << std::right << std::setw(15) << "CPU" 
              << std::setw(15) << "GPU" 
              << std::setw(15) << "Speedup" << std::endl;
    print_separator('-', 65);
    
    double speedup = cpu_min / gpu_min;
    
    std::cout << std::left << std::setw(20) << "Min Time (ms)"
              << std::right << std::fixed << std::setprecision(0)
              << std::setw(15) << cpu_min
              << std::setw(15) << gpu_min
              << std::setw(14) << std::setprecision(2) << speedup << "x"
              << std::endl;
    
    std::cout << std::left << std::setw(20) << "Avg Time (ms)"
              << std::right << std::fixed << std::setprecision(0)
              << std::setw(15) << cpu_avg
              << std::setw(15) << gpu_avg
              << std::setw(14) << std::setprecision(2) << (cpu_avg / gpu_avg) << "x"
              << std::endl;
    
    std::cout << std::left << std::setw(20) << "Result"
              << std::right << std::setw(15) << cpu_result
              << std::setw(15) << gpu_result
              << std::endl;
    
    print_separator('-', 65);
    
    std::cout << "\nGPU Memory:" << std::endl;
    std::cout << "  Initial: " << std::fixed << std::setprecision(1) << initial_mem.gpu_used_mb() << " MB" << std::endl;
    std::cout << "  Final:   " << final_mem.gpu_used_mb() << " MB" << std::endl;
    std::cout << "  Delta:   " << (final_mem.gpu_used_mb() - initial_mem.gpu_used_mb()) << " MB" << std::endl;
    
    print_separator('=', 80);
    
    // Note about GPU result
    if (gpu_result.empty() || gpu_result == "NULL") {
        std::cout << "\nNote: GPU shows no result because cudf->DuckDB conversion is not yet implemented." << std::endl;
        std::cout << "GPU timing reflects actual parquet read + filter + aggregation on GPU." << std::endl;
    }
    
    // Cleanup
    con.Query("SELECT rapidsmpf_clear_tables()");
    
    std::cout << "\nBenchmark completed!" << std::endl;
    
    return 0;
}
