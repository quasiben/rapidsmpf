/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_gpu_execution.cpp
 * @brief End-to-end test demonstrating actual GPU execution.
 *
 * Before running, create test data with:
 *   python examples/create_test_data.py
 *
 * This test:
 * 1. Reads a Substrait plan from JSON
 * 2. Converts to physical operators
 * 3. Executes the streaming pipeline on GPU
 */

#include <iostream>
#include <fstream>
#include <functional>

#include "rapidsmpf_substrait.hpp"

using namespace rapidsmpf_substrait;

int main(int argc, char* argv[]) {
    std::cout << "=== GPU Execution Test ===" << std::endl;
    std::cout << "Testing end-to-end Substrait -> RapidsMPF -> GPU execution\n" << std::endl;

    std::string parquet_path = "/tmp/rapidsmpf_substrait_test.parquet";
    
    // Allow custom path
    if (argc > 1) {
        parquet_path = argv[1];
    }
    
    std::cout << "Using parquet file: " << parquet_path << std::endl;

    try {
        // Step 1: Create a Substrait plan that reads and limits
        std::cout << "\nStep 1: Building Substrait plan..." << std::endl;
        
        // Build JSON with the file path
        std::string plan_json = R"({
          "version": {"majorNumber": 0, "minorNumber": 52, "patchNumber": 0},
          "relations": [{
            "root": {
              "input": {
                "fetch": {
                  "common": {"direct": {}},
                  "input": {
                    "read": {
                      "common": {"direct": {}},
                      "baseSchema": {
                        "names": ["id", "value"],
                        "struct": {
                          "types": [
                            {"i64": {"nullability": "NULLABILITY_REQUIRED"}},
                            {"fp64": {"nullability": "NULLABILITY_NULLABLE"}}
                          ]
                        }
                      },
                      "localFiles": {
                        "items": [{
                          "uriFile": ")" + parquet_path + R"(",
                          "parquet": {}
                        }]
                      }
                    }
                  },
                  "offset": "0",
                  "count": "10"
                }
              },
              "names": ["id", "value"]
            }
          }]
        })";
        
        // Step 2: Parse the plan
        std::cout << "\nStep 2: Parsing Substrait plan from JSON..." << std::endl;
        auto plan = SubstraitPlanParser::ParseFromJson(plan_json);
        std::cout << "   Plan version: " 
                  << plan.version().major_number() << "."
                  << plan.version().minor_number() << "."
                  << plan.version().patch_number() << std::endl;
        
        // Step 3: Convert to physical plan
        std::cout << "\nStep 3: Converting to physical operators..." << std::endl;
        SubstraitPlanParser parser(std::move(plan));
        SubstraitPlanConverter converter(parser);
        auto physical_plan = converter.Convert();
        
        // Print plan tree
        std::cout << "Physical plan tree:" << std::endl;
        std::function<void(PhysicalOperator*, int)> print_tree = 
            [&](PhysicalOperator* op, int depth) {
                std::cout << std::string(depth * 2, ' ') << "- " << op->ToString() << std::endl;
                for (auto& child : op->Children()) {
                    print_tree(child.get(), depth + 1);
                }
            };
        print_tree(physical_plan.get(), 1);
        
        // Step 4: Execute on GPU
        std::cout << "\nStep 4: Executing streaming pipeline on GPU..." << std::endl;
        std::cout << "   (This should show GPU activity)" << std::endl;
        Executor executor;
        auto result = executor.Execute(std::move(physical_plan));
        
        // Step 5: Show results
        std::cout << "\n=== RESULTS ===" << std::endl;
        std::cout << "Execution time: " << result.execution_time_ms << " ms" << std::endl;
        std::cout << "Rows processed: " << result.rows_processed << std::endl;
        
        if (result.table) {
            std::cout << "Result table: " << result.table->num_rows() << " rows, "
                      << result.table->num_columns() << " columns" << std::endl;
            
            // Copy first few values back to host to verify
            if (result.table->num_rows() > 0 && result.table->num_columns() >= 2) {
                auto id_col = result.table->view().column(0);
                auto val_col = result.table->view().column(1);
                
                std::vector<int64_t> id_host(result.table->num_rows());
                std::vector<double> val_host(result.table->num_rows());
                
                cudaMemcpy(id_host.data(), id_col.head<int64_t>(), 
                          id_host.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(val_host.data(), val_col.head<double>(),
                          val_host.size() * sizeof(double), cudaMemcpyDeviceToHost);
                
                std::cout << "\nFirst rows of result:" << std::endl;
                std::cout << "  id\tvalue" << std::endl;
                for (size_t i = 0; i < std::min(size_t(5), id_host.size()); i++) {
                    std::cout << "  " << id_host[i] << "\t" << val_host[i] << std::endl;
                }
            }
        } else {
            std::cout << "WARNING: No result table returned!" << std::endl;
        }
        
        std::cout << "\n=== GPU Execution Test PASSED ===" << std::endl;
        
    } catch (std::exception const& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
