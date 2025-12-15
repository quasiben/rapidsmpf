# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(rapidsmpf_substrait LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Find dependencies
# =============================================================================

# Find rapidsmpf (from sibling cpp directory)
find_package(rapidsmpf REQUIRED
    HINTS ${CMAKE_SOURCE_DIR}/../cpp/build
)

# Find cudf
find_package(cudf REQUIRED)

# Find RMM
find_package(rmm REQUIRED)

# Find protobuf for Substrait plan parsing
find_package(Protobuf REQUIRED)

# Find absl (protobuf dependency)
find_package(absl REQUIRED)

# =============================================================================
# Generate Substrait protobuf files
# =============================================================================

set(SUBSTRAIT_PROTO_DIR ${CMAKE_SOURCE_DIR}/../substrait/proto)
set(GENERATED_PROTO_DIR ${CMAKE_BINARY_DIR}/generated)

file(MAKE_DIRECTORY ${GENERATED_PROTO_DIR})

# Find all .proto files
file(GLOB_RECURSE SUBSTRAIT_PROTOS
    "${SUBSTRAIT_PROTO_DIR}/substrait/*.proto"
)

# Generate protobuf sources
set(GENERATED_PROTO_SOURCES "")
set(GENERATED_PROTO_HEADERS "")

foreach(proto ${SUBSTRAIT_PROTOS})
    get_filename_component(proto_name ${proto} NAME_WE)
    get_filename_component(proto_dir ${proto} DIRECTORY)
    file(RELATIVE_PATH proto_rel_dir ${SUBSTRAIT_PROTO_DIR} ${proto_dir})
    
    set(proto_src "${GENERATED_PROTO_DIR}/${proto_rel_dir}/${proto_name}.pb.cc")
    set(proto_hdr "${GENERATED_PROTO_DIR}/${proto_rel_dir}/${proto_name}.pb.h")
    
    list(APPEND GENERATED_PROTO_SOURCES ${proto_src})
    list(APPEND GENERATED_PROTO_HEADERS ${proto_hdr})
    
    add_custom_command(
        OUTPUT ${proto_src} ${proto_hdr}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${SUBSTRAIT_PROTO_DIR}
            --cpp_out=${GENERATED_PROTO_DIR}
            ${proto}
        DEPENDS ${proto}
        COMMENT "Generating protobuf for ${proto_name}"
    )
endforeach()

# Create a library for generated protobuf sources
add_library(substrait_proto STATIC ${GENERATED_PROTO_SOURCES})
target_include_directories(substrait_proto PUBLIC ${GENERATED_PROTO_DIR})
target_link_libraries(substrait_proto PUBLIC
    protobuf::libprotobuf
    absl::status
    absl::strings
    absl::log_internal_message
    absl::log_internal_check_op
)

# =============================================================================
# rapidsmpf_substrait library
# =============================================================================

set(RAPIDSMPF_SUBSTRAIT_SOURCES
    src/executor.cpp
    src/substrait_plan_parser.cpp
    src/substrait_plan_converter.cpp
    src/physical_operator.cpp
    src/operators/physical_read_rel.cpp
    src/operators/physical_filter_rel.cpp
    src/operators/physical_project_rel.cpp
    src/operators/physical_aggregate_rel.cpp
    src/operators/physical_fetch_rel.cpp
    src/operators/physical_sort_rel.cpp
)

add_library(rapidsmpf_substrait STATIC ${RAPIDSMPF_SUBSTRAIT_SOURCES})

target_include_directories(rapidsmpf_substrait PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${GENERATED_PROTO_DIR}
)

target_link_libraries(rapidsmpf_substrait PUBLIC
    rapidsmpf::rapidsmpf
    cudf::cudf
    rmm::rmm
    substrait_proto
)

# =============================================================================
# Examples
# =============================================================================

add_executable(example_substrait examples/example_substrait.cpp)
target_link_libraries(example_substrait PRIVATE rapidsmpf_substrait)

add_executable(example_from_json examples/example_from_json.cpp)
target_link_libraries(example_from_json PRIVATE rapidsmpf_substrait)

add_executable(test_gpu_execution examples/test_gpu_execution.cpp)
target_link_libraries(test_gpu_execution PRIVATE rapidsmpf_substrait)

# =============================================================================
# Installation
# =============================================================================

install(TARGETS rapidsmpf_substrait substrait_proto
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY src/include/
    DESTINATION include/rapidsmpf_substrait
)

