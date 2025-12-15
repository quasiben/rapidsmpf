/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_read_rel.hpp"

#include <sstream>
#include <stdexcept>

// Undefine DEBUG to avoid conflict
#ifdef DEBUG
#undef DEBUG
#endif

#include <cudf/io/parquet.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf_substrait {

PhysicalReadRel::PhysicalReadRel(
    std::string table_name,
    std::vector<std::string> file_paths,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> column_types,
    std::vector<int32_t> projected_columns,
    std::optional<ExpressionInfo> filter_expr
) : PhysicalOperator(
        PhysicalOperatorType::READ,
        std::move(column_types),
        0  // Unknown cardinality
    ),
    table_name_(std::move(table_name)),
    file_paths_(std::move(file_paths)),
    column_names_(std::move(column_names)),
    projected_columns_(std::move(projected_columns)),
    filter_expr_(std::move(filter_expr)) {}

rapidsmpf::streaming::Node PhysicalReadRel::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> /* ch_in */,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    if (file_paths_.empty()) {
        throw std::runtime_error(
            "No file paths configured for table scan on table '" + table_name_ + "'. "
            "Provide local_files in the Substrait plan or use a table registry."
        );
    }

    // Build the list of columns to read
    std::vector<std::string> columns_to_read;
    if (!projected_columns_.empty()) {
        for (auto idx : projected_columns_) {
            if (idx >= 0 && static_cast<size_t>(idx) < column_names_.size()) {
                columns_to_read.push_back(column_names_[idx]);
            }
        }
    } else if (!column_names_.empty()) {
        columns_to_read = column_names_;
    }

    // Build parquet reader options
    auto source = cudf::io::source_info(file_paths_);
    auto builder = cudf::io::parquet_reader_options::builder(source);
    
    if (!columns_to_read.empty()) {
        builder.columns(columns_to_read);
    }
    
    auto options = builder.build();

    // Create the parquet reader node
    // TODO: Add filter pushdown using filter_expr_ when cudf AST is integrated
    return rapidsmpf::streaming::node::read_parquet(
        ctx,
        ch_out,
        1,  // num_producers
        options,
        1024 * 1024,  // 1M rows per chunk
        nullptr  // No filter pushdown for now
    );
}

std::string PhysicalReadRel::ToString() const {
    std::ostringstream oss;
    oss << "READ(";
    if (!table_name_.empty()) {
        oss << "table=" << table_name_;
    } else if (!file_paths_.empty()) {
        oss << "files=" << file_paths_.size();
    }
    oss << ", cols=" << column_names_.size();
    if (!projected_columns_.empty()) {
        oss << ", projected=" << projected_columns_.size();
    }
    if (filter_expr_.has_value()) {
        oss << ", filtered";
    }
    oss << ")";
    return oss.str();
}

}  // namespace rapidsmpf_substrait

