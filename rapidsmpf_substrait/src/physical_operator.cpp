/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/physical_operator.hpp"

#include <sstream>

namespace rapidsmpf_substrait {

std::string PhysicalOperatorTypeToString(PhysicalOperatorType type) {
    switch (type) {
        case PhysicalOperatorType::READ:
            return "READ";
        case PhysicalOperatorType::FILTER:
            return "FILTER";
        case PhysicalOperatorType::PROJECT:
            return "PROJECT";
        case PhysicalOperatorType::AGGREGATE:
            return "AGGREGATE";
        case PhysicalOperatorType::SORT:
            return "SORT";
        case PhysicalOperatorType::FETCH:
            return "FETCH";
        case PhysicalOperatorType::JOIN:
            return "JOIN";
        case PhysicalOperatorType::SET:
            return "SET";
        default:
            return "UNKNOWN";
    }
}

PhysicalOperator::PhysicalOperator(
    PhysicalOperatorType type,
    std::vector<cudf::data_type> output_types,
    cudf::size_type estimated_cardinality
) : type_(type),
    output_types_(std::move(output_types)),
    estimated_cardinality_(estimated_cardinality) {}

std::string PhysicalOperator::ToString() const {
    std::ostringstream oss;
    oss << PhysicalOperatorTypeToString(type_);
    oss << " (cols=" << output_types_.size();
    oss << ", est_rows=" << estimated_cardinality_ << ")";
    return oss.str();
}

}  // namespace rapidsmpf_substrait

