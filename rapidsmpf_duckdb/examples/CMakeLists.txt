# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# Common include directories for examples
set(EXAMPLE_INCLUDES ${CMAKE_SOURCE_DIR}/src/include)

# Common libraries for examples
set(EXAMPLE_LIBS duckdb_static)

# Example: TPCH queries using rapidsmpf_duckdb extension
add_executable(example_tpch example_tpch.cpp)
target_link_libraries(example_tpch ${EXAMPLE_LIBS})
target_include_directories(example_tpch PRIVATE ${EXAMPLE_INCLUDES})

# Example: Registered tables with GPU execution
add_executable(example_registered_tables example_registered_tables.cpp)
target_link_libraries(example_registered_tables ${EXAMPLE_LIBS})
target_include_directories(example_registered_tables PRIVATE ${EXAMPLE_INCLUDES})

# Benchmark: GPU vs CPU performance comparison
add_executable(benchmark_gpu_vs_cpu benchmark_gpu_vs_cpu.cpp)
target_link_libraries(benchmark_gpu_vs_cpu ${EXAMPLE_LIBS})
target_include_directories(benchmark_gpu_vs_cpu PRIVATE ${EXAMPLE_INCLUDES})

