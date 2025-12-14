/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>
#include <memory>
#include <string>
#include <vector>

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>

#include "duckdb/planner/expression.hpp"

namespace rapidsmpf_duckdb {

/**
 * @brief Converts DuckDB expressions to cudf AST expressions for filter pushdown.
 *
 * This class handles the conversion of DuckDB's bound expressions (after planning)
 * to cudf's AST expression tree which can be pushed down to parquet reading.
 *
 * Supported expressions:
 * - Comparison: =, <, >, <=, >=, != (BoundComparisonExpression)
 * - Conjunction: AND, OR (BoundConjunctionExpression)
 * - Column reference (BoundColumnRefExpression)
 * - Constants: integers, floats, strings, dates (BoundConstantExpression)
 *
 * Example DuckDB expression:
 *   l_quantity > 25 AND l_shipdate < '1998-09-02'
 *
 * Converts to cudf AST:
 *   operation(AND,
 *     operation(GREATER, column_name_reference("l_quantity"), literal(25)),
 *     operation(LESS, column_name_reference("l_shipdate"), literal(timestamp)))
 */
class ExpressionConverter {
  public:
    /**
     * @brief Create a converter with a CUDA stream for scalar creation.
     */
    explicit ExpressionConverter(rmm::cuda_stream_view stream);

    /**
     * @brief Convert a DuckDB expression to a rapidsmpf Filter.
     *
     * @param expr The DuckDB expression to convert.
     * @param column_names Column names in the table (for name lookup).
     * @return A Filter that can be passed to read_parquet.
     * @throws std::runtime_error if the expression cannot be converted.
     */
    std::unique_ptr<rapidsmpf::streaming::Filter> Convert(
        duckdb::Expression& expr,
        std::vector<std::string> const& column_names
    );

    /**
     * @brief Check if an expression can be converted to cudf AST.
     *
     * @param expr The expression to check.
     * @return true if the expression is supported for pushdown.
     */
    static bool CanConvert(duckdb::Expression const& expr);

  private:
    /**
     * @brief Recursively convert an expression.
     *
     * Stores all created objects in owner_ for lifetime management.
     * Returns a pointer to the created cudf::ast::expression.
     */
    cudf::ast::expression* ConvertImpl(
        duckdb::Expression& expr,
        std::vector<std::string> const& column_names
    );

    /**
     * @brief Convert a comparison expression (=, <, >, etc.)
     */
    cudf::ast::expression* ConvertComparison(
        duckdb::Expression& expr,
        std::vector<std::string> const& column_names
    );

    /**
     * @brief Convert a conjunction expression (AND, OR)
     */
    cudf::ast::expression* ConvertConjunction(
        duckdb::Expression& expr,
        std::vector<std::string> const& column_names
    );

    /**
     * @brief Convert a column reference expression
     */
    cudf::ast::expression* ConvertColumnRef(
        duckdb::Expression& expr,
        std::vector<std::string> const& column_names
    );

    /**
     * @brief Convert a constant/literal expression
     */
    cudf::ast::expression* ConvertConstant(duckdb::Expression& expr);

    /**
     * @brief Get the cudf AST operator for a DuckDB expression type.
     */
    static cudf::ast::ast_operator GetOperator(duckdb::ExpressionType type);

    rmm::cuda_stream_view stream_;
    std::vector<std::any> owner_;  ///< Owns all created objects
};

}  // namespace rapidsmpf_duckdb


