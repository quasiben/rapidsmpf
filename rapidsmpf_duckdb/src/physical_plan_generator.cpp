/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/physical_plan_generator.hpp"
#include "include/operators/physical_table_scan.hpp"
#include "include/operators/physical_filter.hpp"
#include "include/operators/physical_projection.hpp"
#include "include/operators/physical_aggregate.hpp"
#include "include/operators/physical_limit.hpp"
#include "include/operators/physical_order.hpp"

namespace rapidsmpf_duckdb {

PhysicalPlanGenerator::PhysicalPlanGenerator(duckdb::ClientContext& context)
    : context_(context) {}

std::unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(
    std::unique_ptr<duckdb::LogicalOperator> op
) {
    // Resolve types on the logical plan if not already done
    op->ResolveOperatorTypes();
    return CreatePlan(*op);
}

std::unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(
    duckdb::LogicalOperator& op
) {
    switch (op.type) {
        case duckdb::LogicalOperatorType::LOGICAL_GET:
            return CreatePlan(op.Cast<duckdb::LogicalGet>());

        case duckdb::LogicalOperatorType::LOGICAL_FILTER:
            return CreatePlan(op.Cast<duckdb::LogicalFilter>());

        case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
            return CreatePlan(op.Cast<duckdb::LogicalProjection>());

        case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
            return CreatePlan(op.Cast<duckdb::LogicalAggregate>());

        case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
            return CreatePlan(op.Cast<duckdb::LogicalLimit>());

        case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
            return CreatePlan(op.Cast<duckdb::LogicalOrder>());

        default:
            throw std::runtime_error(
                "Unsupported logical operator type: " +
                duckdb::LogicalOperatorToString(op.type) +
                ". Currently supported: GET, FILTER, PROJECTION, AGGREGATE, LIMIT, ORDER BY."
            );
    }
}

std::unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(
    duckdb::LogicalGet& op
) {
    return std::make_unique<PhysicalTableScan>(op);
}

std::unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(
    duckdb::LogicalFilter& op
) {
    // Recursively create the child plan first
    if (op.children.empty()) {
        throw std::runtime_error("LogicalFilter has no child operator");
    }
    auto child = CreatePlan(*op.children[0]);
    return std::make_unique<PhysicalFilter>(op, std::move(child));
}

std::unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(
    duckdb::LogicalProjection& op
) {
    // Recursively create the child plan first
    if (op.children.empty()) {
        throw std::runtime_error("LogicalProjection has no child operator");
    }
    auto child = CreatePlan(*op.children[0]);
    return std::make_unique<PhysicalProjection>(op, std::move(child));
}

std::unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(
    duckdb::LogicalAggregate& op
) {
    // Recursively create the child plan first
    if (op.children.empty()) {
        throw std::runtime_error("LogicalAggregate has no child operator");
    }
    auto child = CreatePlan(*op.children[0]);
    return std::make_unique<PhysicalAggregate>(op, std::move(child));
}

std::unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(
    duckdb::LogicalLimit& op
) {
    // Recursively create the child plan first
    if (op.children.empty()) {
        throw std::runtime_error("LogicalLimit has no child operator");
    }
    auto child = CreatePlan(*op.children[0]);
    return std::make_unique<PhysicalLimit>(op, std::move(child));
}

std::unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(
    duckdb::LogicalOrder& op
) {
    // Recursively create the child plan first
    if (op.children.empty()) {
        throw std::runtime_error("LogicalOrder has no child operator");
    }
    auto child = CreatePlan(*op.children[0]);
    return std::make_unique<PhysicalOrder>(op, std::move(child));
}

}  // namespace rapidsmpf_duckdb
