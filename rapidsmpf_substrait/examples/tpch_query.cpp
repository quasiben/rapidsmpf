/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file tpch_query.cpp
 * @brief TPC-H query example using Substrait -> RapidsMPF
 *
 * Query:
 *   SELECT AVG(l_extendedprice) as avg_price 
 *   FROM lineitem 
 *   WHERE l_quantity > 25
 *
 * Uses lineitem data from /raid/rapidsmpf/data/tpch/scale-1000
 */

#include <iostream>
#include <fstream>
#include <functional>
#include <filesystem>

#include "rapidsmpf_substrait.hpp"

using namespace rapidsmpf_substrait;
namespace fs = std::filesystem;

// Get all parquet files in a directory
std::vector<std::string> get_parquet_files(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".parquet") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char* argv[]) {
    std::cout << "=== TPC-H Query Example ===" << std::endl;
    std::cout << "Query: SELECT AVG(l_extendedprice) as avg_price FROM lineitem WHERE l_quantity > 25\n" << std::endl;

    std::string lineitem_dir = "/raid/rapidsmpf/data/tpch/scale-1000/lineitem";
    
    // Allow custom path
    if (argc > 1) {
        lineitem_dir = argv[1];
    }
    
    std::cout << "Data directory: " << lineitem_dir << std::endl;

    try {
        // Get all parquet files
        auto parquet_files = get_parquet_files(lineitem_dir);
        std::cout << "Found " << parquet_files.size() << " parquet files" << std::endl;
        
        if (parquet_files.empty()) {
            std::cerr << "No parquet files found!" << std::endl;
            return 1;
        }
        
        // For this demo, use first few files to keep runtime reasonable
        size_t max_files = 4;  // Use 4 files for demo
        if (parquet_files.size() > max_files) {
            std::cout << "Using first " << max_files << " files for demo..." << std::endl;
            parquet_files.resize(max_files);
        }

        // Build the Substrait plan JSON
        // This represents:
        //   SELECT AVG(l_extendedprice) 
        //   FROM lineitem 
        //   WHERE l_quantity > 25
        //
        // lineitem schema (relevant columns):
        //   4: l_quantity (float64)
        //   5: l_extendedprice (float64)
        
        // Build file list for JSON
        std::string files_json;
        for (size_t i = 0; i < parquet_files.size(); i++) {
            if (i > 0) files_json += ",\n                    ";
            files_json += R"({"uriFile": ")" + parquet_files[i] + R"(", "parquet": {}})";
        }

        std::string plan_json = R"({
          "version": {"majorNumber": 0, "minorNumber": 52, "patchNumber": 0},
          "extensionUris": [{
            "extensionUriAnchor": 1,
            "uri": "https://github.com/substrait-io/substrait/blob/main/extensions/functions_arithmetic.yaml"
          }, {
            "extensionUriAnchor": 2,
            "uri": "https://github.com/substrait-io/substrait/blob/main/extensions/functions_comparison.yaml"
          }],
          "extensions": [{
            "extensionFunction": {
              "extensionUriReference": 1,
              "functionAnchor": 0,
              "name": "avg:fp64"
            }
          }, {
            "extensionFunction": {
              "extensionUriReference": 2,
              "functionAnchor": 1,
              "name": "gt:fp64_fp64"
            }
          }],
          "relations": [{
            "root": {
              "input": {
                "aggregate": {
                  "common": {"direct": {}},
                  "input": {
                    "filter": {
                      "common": {"direct": {}},
                      "input": {
                        "read": {
                          "common": {"direct": {}},
                          "baseSchema": {
                            "names": ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", 
                                      "l_quantity", "l_extendedprice", "l_discount", "l_tax",
                                      "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate",
                                      "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
                            "struct": {
                              "types": [
                                {"i64": {"nullability": "NULLABILITY_REQUIRED"}},
                                {"i64": {"nullability": "NULLABILITY_REQUIRED"}},
                                {"i64": {"nullability": "NULLABILITY_REQUIRED"}},
                                {"i64": {"nullability": "NULLABILITY_REQUIRED"}},
                                {"fp64": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"fp64": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"fp64": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"fp64": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"string": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"string": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"date": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"date": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"date": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"string": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"string": {"nullability": "NULLABILITY_NULLABLE"}},
                                {"string": {"nullability": "NULLABILITY_NULLABLE"}}
                              ]
                            }
                          },
                          "projection": {
                            "select": {
                              "structItems": [
                                {"field": 4},
                                {"field": 5}
                              ]
                            }
                          },
                          "localFiles": {
                            "items": [
                    )" + files_json + R"(
                            ]
                          }
                        }
                      },
                      "condition": {
                        "scalarFunction": {
                          "functionReference": 1,
                          "outputType": {"bool": {"nullability": "NULLABILITY_NULLABLE"}},
                          "arguments": [{
                            "value": {
                              "selection": {
                                "directReference": {"structField": {"field": 0}},
                                "rootReference": {}
                              }
                            }
                          }, {
                            "value": {
                              "literal": {"fp64": 25.0}
                            }
                          }]
                        }
                      }
                    }
                  },
                  "groupings": [],
                  "measures": [{
                    "measure": {
                      "functionReference": 0,
                      "phase": "AGGREGATION_PHASE_INITIAL_TO_RESULT",
                      "outputType": {"fp64": {"nullability": "NULLABILITY_NULLABLE"}},
                      "arguments": [{
                        "value": {
                          "selection": {
                            "directReference": {"structField": {"field": 1}},
                            "rootReference": {}
                          }
                        }
                      }]
                    }
                  }]
                }
              },
              "names": ["avg_price"]
            }
          }]
        })";
        
        // Parse and convert the plan
        std::cout << "\nParsing Substrait plan..." << std::endl;
        auto plan = SubstraitPlanParser::ParseFromJson(plan_json);
        std::cout << "Plan version: " 
                  << plan.version().major_number() << "."
                  << plan.version().minor_number() << "."
                  << plan.version().patch_number() << std::endl;
        
        // Convert to physical plan
        std::cout << "\nConverting to physical operators..." << std::endl;
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
        
        // Execute on GPU
        std::cout << "\nExecuting on GPU..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        Executor executor;
        auto result = executor.Execute(std::move(physical_plan));
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Show results
        std::cout << "\n=== RESULTS ===" << std::endl;
        std::cout << "Execution time: " << elapsed << " ms" << std::endl;
        std::cout << "Rows processed: " << result.rows_processed << std::endl;
        
        if (result.table && result.table->num_rows() > 0) {
            auto avg_col = result.table->view().column(0);
            double avg_value;
            cudaMemcpy(&avg_value, avg_col.head<double>(), sizeof(double), cudaMemcpyDeviceToHost);
            
            std::cout << "\nQuery Result:" << std::endl;
            std::cout << "  AVG(l_extendedprice) WHERE l_quantity > 25 = " << avg_value << std::endl;
        } else {
            std::cout << "No results returned" << std::endl;
        }
        
        std::cout << "\n=== Query Complete ===" << std::endl;
        
    } catch (std::exception const& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

