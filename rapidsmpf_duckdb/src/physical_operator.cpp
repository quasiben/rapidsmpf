/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/physical_operator.hpp"

#include "duckdb/common/enums/physical_operator_type.hpp"

namespace rapidsmpf_duckdb {

PhysicalOperator::PhysicalOperator(
    duckdb::PhysicalOperatorType type,
    std::vector<duckdb::LogicalType> types,
    duckdb::idx_t estimated_cardinality
)
    : type_(type)
    , types_(std::move(types))
    , estimated_cardinality_(estimated_cardinality) {}

std::string PhysicalOperator::GetName() const {
    return "RAPIDSMPF_OPERATOR";
}

void PhysicalOperator::AddChild(std::unique_ptr<PhysicalOperator> child) {
    children_.push_back(std::move(child));
}

std::string PhysicalOperator::ToString(int indent) const {
    std::string result(indent, ' ');
    result += GetName();
    result += " (estimated_cardinality=" + std::to_string(estimated_cardinality_) + ")";
    result += "\n";
    
    for (auto const& child : children_) {
        result += child->ToString(indent + 2);
    }
    
    return result;
}

}  // namespace rapidsmpf_duckdb




