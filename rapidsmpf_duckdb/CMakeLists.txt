# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
#
# rapidsmpf_duckdb - DuckDB extension for executing queries via rapidsmpf
# 
# This extension converts DuckDB's LogicalOperator tree into rapidsmpf
# streaming operators for GPU-accelerated execution.
# =============================================================================

cmake_minimum_required(VERSION 3.30.4)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set extension name
set(TARGET_NAME rapidsmpf_duckdb)
project(${TARGET_NAME} LANGUAGES CXX CUDA)

# Find required packages (guard against duplicate finds when building as sub-extension)
if(NOT TARGET cudf::cudf)
  find_package(cudf REQUIRED CONFIG)
endif()
if(NOT TARGET rapidsmpf::rapidsmpf)
  find_package(rapidsmpf REQUIRED CONFIG)
endif()

# CUDA settings
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

# Include directories
include_directories(src/include)

# Extension sources
set(EXTENSION_SOURCES
  src/rapidsmpf_extension.cpp
  src/physical_operator.cpp
  src/physical_plan_generator.cpp
  src/executor.cpp
  src/table_registry.cpp
  src/operators/physical_table_scan.cpp
  src/operators/physical_projection.cpp
  src/operators/physical_filter.cpp
  src/operators/physical_aggregate.cpp
  src/operators/physical_limit.cpp
  src/operators/physical_order.cpp
)

# Build the extension using DuckDB's extension build system
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link libraries
target_link_libraries(${EXTENSION_NAME}
  cudf::cudf
  rmm::rmm
  rapidsmpf::rapidsmpf
)
target_link_libraries(${LOADABLE_EXTENSION_NAME}
  cudf::cudf
  rmm::rmm
  rapidsmpf::rapidsmpf
)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

# Build examples
add_subdirectory(examples)
