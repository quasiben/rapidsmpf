/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/substrait_plan_parser.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <google/protobuf/util/json_util.h>

namespace rapidsmpf_substrait {

substrait::Plan SubstraitPlanParser::ParseFromBinary(std::string const& data) {
    substrait::Plan plan;
    if (!plan.ParseFromString(data)) {
        throw std::runtime_error("Failed to parse Substrait plan from binary format");
    }
    return plan;
}

substrait::Plan SubstraitPlanParser::ParseFromJson(std::string const& json) {
    substrait::Plan plan;
    auto status = google::protobuf::util::JsonStringToMessage(json, &plan);
    if (!status.ok()) {
        throw std::runtime_error(
            "Failed to parse Substrait plan from JSON: " + std::string(status.message())
        );
    }
    return plan;
}

substrait::Plan SubstraitPlanParser::ParseFromFile(std::string const& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Try to detect format: JSON starts with '{' or whitespace then '{'
    size_t first_char = content.find_first_not_of(" \t\n\r");
    if (first_char != std::string::npos && content[first_char] == '{') {
        return ParseFromJson(content);
    }

    // Otherwise, try binary
    return ParseFromBinary(content);
}

SubstraitPlanParser::SubstraitPlanParser(substrait::Plan plan)
    : plan_(std::move(plan)) {
    BuildFunctionMap();
}

void SubstraitPlanParser::BuildFunctionMap() {
    // Build map from extension function declarations
    for (auto const& ext : plan_.extensions()) {
        if (ext.has_extension_function()) {
            auto const& func = ext.extension_function();
            functions_map_[func.function_anchor()] = func.name();
        }
    }
}

std::string SubstraitPlanParser::FindFunction(uint32_t function_anchor) const {
    auto it = functions_map_.find(function_anchor);
    if (it == functions_map_.end()) {
        throw std::runtime_error(
            "Function not found for anchor: " + std::to_string(function_anchor)
        );
    }
    return it->second;
}

substrait::RelRoot const& SubstraitPlanParser::GetRootRelation() const {
    if (plan_.relations().empty()) {
        throw std::runtime_error("Substrait plan has no relations");
    }
    auto const& plan_rel = plan_.relations(0);
    if (!plan_rel.has_root()) {
        throw std::runtime_error("Substrait plan relation is not a root");
    }
    return plan_rel.root();
}

}  // namespace rapidsmpf_substrait

