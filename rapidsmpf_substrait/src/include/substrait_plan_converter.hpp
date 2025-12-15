/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cudf/types.hpp>

#include "substrait/algebra.pb.h"
#include "substrait/plan.pb.h"

#include "physical_operator.hpp"
#include "substrait_plan_parser.hpp"

namespace rapidsmpf_substrait {

/**
 * @brief Expression information extracted from Substrait expressions.
 *
 * Used to pass expression details to physical operators.
 */
struct ExpressionInfo {
    enum class Type {
        LITERAL,
        FIELD_REFERENCE,
        SCALAR_FUNCTION,
        CAST,
        IF_THEN,
        UNKNOWN
    };

    Type type = Type::UNKNOWN;

    /// For literals: the value as a string
    std::string literal_value;

    /// For field references: the field index (0-based)
    int32_t field_index = -1;

    /// For scalar functions: the function name
    std::string function_name;

    /// For scalar functions: child expressions
    std::vector<ExpressionInfo> arguments;

    /// The output data type (if known)
    cudf::data_type output_type{cudf::type_id::EMPTY};
};

/**
 * @brief Aggregate function information.
 */
struct AggregateInfo {
    std::string function_name;
    std::vector<int32_t> argument_indices;
    cudf::data_type output_type{cudf::type_id::EMPTY};
    bool is_distinct = false;
};

/**
 * @brief Sort field information.
 */
struct SortFieldInfo {
    int32_t field_index;
    bool ascending;
    bool nulls_first;
};

/**
 * @brief Converts Substrait plans to rapidsmpf physical operators.
 *
 * This class traverses a Substrait plan and creates the corresponding
 * tree of PhysicalOperator nodes that can be executed by the Executor.
 */
class SubstraitPlanConverter {
  public:
    /**
     * @brief Construct a converter for a parsed plan.
     *
     * @param parser The plan parser containing the Substrait plan.
     */
    explicit SubstraitPlanConverter(SubstraitPlanParser const& parser);

    /**
     * @brief Convert the plan to a physical operator tree.
     *
     * @return The root physical operator.
     */
    [[nodiscard]] std::unique_ptr<PhysicalOperator> Convert();

    /**
     * @brief Convert a single relation to a physical operator.
     *
     * @param rel The Substrait relation.
     * @return The physical operator.
     */
    [[nodiscard]] std::unique_ptr<PhysicalOperator> ConvertRelation(
        substrait::Rel const& rel
    );

  private:
    SubstraitPlanParser const& parser_;
    std::vector<std::string> output_names_;

    // Relation converters
    [[nodiscard]] std::unique_ptr<PhysicalOperator> ConvertReadRel(
        substrait::ReadRel const& read
    );
    [[nodiscard]] std::unique_ptr<PhysicalOperator> ConvertFilterRel(
        substrait::FilterRel const& filter
    );
    [[nodiscard]] std::unique_ptr<PhysicalOperator> ConvertProjectRel(
        substrait::ProjectRel const& project
    );
    [[nodiscard]] std::unique_ptr<PhysicalOperator> ConvertAggregateRel(
        substrait::AggregateRel const& aggregate
    );
    [[nodiscard]] std::unique_ptr<PhysicalOperator> ConvertSortRel(
        substrait::SortRel const& sort
    );
    [[nodiscard]] std::unique_ptr<PhysicalOperator> ConvertFetchRel(
        substrait::FetchRel const& fetch
    );

    // Expression converters
    [[nodiscard]] ExpressionInfo ConvertExpression(
        substrait::Expression const& expr
    );

    // Type converters
    [[nodiscard]] static cudf::data_type ConvertType(substrait::Type const& type);
    [[nodiscard]] static std::vector<cudf::data_type> ConvertSchema(
        substrait::NamedStruct const& schema
    );

    // Helper to get function name from anchor
    [[nodiscard]] std::string GetFunctionName(uint32_t anchor) const;

    // Helper to remove extension suffix from function name (e.g., "add:i32_i32" -> "add")
    [[nodiscard]] static std::string RemoveExtension(std::string const& name);
};

}  // namespace rapidsmpf_substrait

