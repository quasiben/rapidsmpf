/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file example_substrait.cpp
 * @brief Example demonstrating rapidsmpf_substrait usage.
 *
 * This example shows how to:
 * 1. Create a Substrait plan programmatically
 * 2. Convert it to a physical plan
 * 3. Execute it on GPU using rapidsmpf
 */

#include <iostream>
#include <vector>

#include "rapidsmpf_substrait.hpp"

using namespace rapidsmpf_substrait;

int main(int argc, char* argv[]) {
    std::cout << "=== rapidsmpf_substrait Example ===" << std::endl;

    // Example 1: Create a simple plan programmatically
    // This demonstrates creating physical operators directly
    {
        std::cout << "\n--- Example 1: Direct Physical Plan ---" << std::endl;

        // Create a read operator for parquet files
        std::vector<std::string> file_paths = {"/path/to/data.parquet"};
        std::vector<std::string> column_names = {"id", "value", "category"};
        std::vector<cudf::data_type> column_types = {
            cudf::data_type{cudf::type_id::INT64},
            cudf::data_type{cudf::type_id::FLOAT64},
            cudf::data_type{cudf::type_id::STRING}
        };

        auto read_op = std::make_unique<PhysicalReadRel>(
            "",  // No table name, using file paths directly
            file_paths,
            column_names,
            column_types,
            std::vector<int32_t>{0, 1}  // Project only id and value
        );

        std::cout << "Created: " << read_op->ToString() << std::endl;

        // Note: Execution would require actual parquet files
        // Executor executor;
        // auto result = executor.Execute(std::move(read_op));
    }

    // Example 2: Parse and execute from JSON
    if (argc > 1) {
        std::cout << "\n--- Example 2: Execute from File ---" << std::endl;
        std::string file_path = argv[1];

        try {
            auto result = ExecuteFromFile(file_path);
            std::cout << "Execution completed!" << std::endl;
            std::cout << "  Rows processed: " << result.rows_processed << std::endl;
            std::cout << "  Execution time: " << result.execution_time_ms << " ms" << std::endl;
            
            if (result.table) {
                std::cout << "  Result columns: " << result.table->num_columns() << std::endl;
                std::cout << "  Result rows: " << result.table->num_rows() << std::endl;
            }
        } catch (std::exception const& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "\nUsage: " << argv[0] << " <substrait_plan.json>" << std::endl;
        std::cout << "Pass a Substrait plan file to execute it." << std::endl;
    }

    std::cout << "\n=== Example Complete ===" << std::endl;
    return 0;
}

