/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file example_from_json.cpp
 * @brief Example demonstrating Substrait JSON plan execution.
 *
 * This example shows how to execute a Substrait plan from JSON format.
 */

#include <iostream>
#include <string>

#include "rapidsmpf_substrait.hpp"

using namespace rapidsmpf_substrait;

// Example Substrait plan in JSON format
// This represents: SELECT * FROM parquet_scan('/path/to/data.parquet') LIMIT 10
const char* EXAMPLE_PLAN_JSON = R"({
  "version": {
    "majorNumber": 0,
    "minorNumber": 52,
    "patchNumber": 0
  },
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
                  "uriFile": "/tmp/test_data.parquet",
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

int main() {
    std::cout << "=== Substrait JSON Example ===" << std::endl;

    try {
        // Parse the JSON plan
        std::cout << "\n1. Parsing Substrait plan from JSON..." << std::endl;
        auto plan = SubstraitPlanParser::ParseFromJson(EXAMPLE_PLAN_JSON);
        std::cout << "   Plan version: " 
                  << plan.version().major_number() << "."
                  << plan.version().minor_number() << "."
                  << plan.version().patch_number() << std::endl;

        // Create parser and converter
        std::cout << "\n2. Converting to physical plan..." << std::endl;
        SubstraitPlanParser parser(std::move(plan));
        SubstraitPlanConverter converter(parser);
        
        auto physical_plan = converter.Convert();
        std::cout << "   Root operator: " << physical_plan->ToString() << std::endl;

        // Print the plan tree
        std::cout << "\n3. Physical plan tree:" << std::endl;
        std::function<void(PhysicalOperator*, int)> print_tree = 
            [&](PhysicalOperator* op, int depth) {
                std::cout << std::string(depth * 2, ' ') << "- " << op->ToString() << std::endl;
                for (auto& child : op->Children()) {
                    print_tree(child.get(), depth + 1);
                }
            };
        print_tree(physical_plan.get(), 1);

        // Note: Execution requires actual parquet files
        std::cout << "\n4. To execute, provide a valid parquet file path in the plan." << std::endl;

        // Uncomment to execute (requires valid parquet file):
        // Executor executor;
        // auto result = executor.Execute(std::move(physical_plan));
        // std::cout << "Result rows: " << result.rows_processed << std::endl;

    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n=== Example Complete ===" << std::endl;
    return 0;
}

