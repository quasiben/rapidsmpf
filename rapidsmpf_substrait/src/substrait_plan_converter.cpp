/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/substrait_plan_converter.hpp"

#include <stdexcept>

#include "include/operators/physical_read_rel.hpp"
#include "include/operators/physical_filter_rel.hpp"
#include "include/operators/physical_project_rel.hpp"
#include "include/operators/physical_aggregate_rel.hpp"
#include "include/operators/physical_sort_rel.hpp"
#include "include/operators/physical_fetch_rel.hpp"

namespace rapidsmpf_substrait {

SubstraitPlanConverter::SubstraitPlanConverter(SubstraitPlanParser const& parser)
    : parser_(parser) {}

std::unique_ptr<PhysicalOperator> SubstraitPlanConverter::Convert() {
    auto const& root = parser_.GetRootRelation();

    // Store output names for later use
    for (auto const& name : root.names()) {
        output_names_.push_back(name);
    }

    return ConvertRelation(root.input());
}

std::unique_ptr<PhysicalOperator> SubstraitPlanConverter::ConvertRelation(
    substrait::Rel const& rel
) {
    switch (rel.rel_type_case()) {
        case substrait::Rel::kRead:
            return ConvertReadRel(rel.read());
        case substrait::Rel::kFilter:
            return ConvertFilterRel(rel.filter());
        case substrait::Rel::kProject:
            return ConvertProjectRel(rel.project());
        case substrait::Rel::kAggregate:
            return ConvertAggregateRel(rel.aggregate());
        case substrait::Rel::kSort:
            return ConvertSortRel(rel.sort());
        case substrait::Rel::kFetch:
            return ConvertFetchRel(rel.fetch());
        default:
            throw std::runtime_error(
                "Unsupported Substrait relation type: " +
                std::to_string(static_cast<int>(rel.rel_type_case()))
            );
    }
}

std::unique_ptr<PhysicalOperator> SubstraitPlanConverter::ConvertReadRel(
    substrait::ReadRel const& read
) {
    // Extract table/file information
    std::string table_name;
    std::vector<std::string> file_paths;

    if (read.has_named_table()) {
        // Named table reference
        if (!read.named_table().names().empty()) {
            table_name = read.named_table().names(0);
        }
    } else if (read.has_local_files()) {
        // Local file paths
        for (auto const& item : read.local_files().items()) {
            if (item.has_uri_file()) {
                file_paths.push_back(item.uri_file());
            } else if (item.has_uri_path()) {
                file_paths.push_back(item.uri_path());
            }
        }
    }

    // Extract schema
    std::vector<std::string> column_names;
    std::vector<cudf::data_type> column_types;
    if (read.has_base_schema()) {
        for (auto const& name : read.base_schema().names()) {
            column_names.push_back(name);
        }
        column_types = ConvertSchema(read.base_schema());
    }

    // Extract projection (column selection)
    std::vector<int32_t> projected_columns;
    if (read.has_projection()) {
        for (auto const& item : read.projection().select().struct_items()) {
            projected_columns.push_back(item.field());
        }
    }

    // Extract filter (if any)
    std::optional<ExpressionInfo> filter_expr;
    if (read.has_filter()) {
        filter_expr = ConvertExpression(read.filter());
    }

    return std::make_unique<PhysicalReadRel>(
        table_name,
        file_paths,
        column_names,
        column_types,
        projected_columns,
        filter_expr
    );
}

std::unique_ptr<PhysicalOperator> SubstraitPlanConverter::ConvertFilterRel(
    substrait::FilterRel const& filter
) {
    // Convert the input relation first
    auto child = ConvertRelation(filter.input());

    // Convert the filter condition
    auto condition = ConvertExpression(filter.condition());

    return std::make_unique<PhysicalFilterRel>(
        std::move(child),
        std::move(condition)
    );
}

std::unique_ptr<PhysicalOperator> SubstraitPlanConverter::ConvertProjectRel(
    substrait::ProjectRel const& project
) {
    // Convert the input relation first
    auto child = ConvertRelation(project.input());

    // Convert projection expressions
    std::vector<ExpressionInfo> expressions;
    for (auto const& expr : project.expressions()) {
        expressions.push_back(ConvertExpression(expr));
    }

    return std::make_unique<PhysicalProjectRel>(
        std::move(child),
        std::move(expressions)
    );
}

std::unique_ptr<PhysicalOperator> SubstraitPlanConverter::ConvertAggregateRel(
    substrait::AggregateRel const& aggregate
) {
    // Convert the input relation first
    auto child = ConvertRelation(aggregate.input());

    // Convert grouping expressions using expression_references (new API)
    std::vector<int32_t> group_by_columns;
    for (auto const& grouping : aggregate.groupings()) {
        // Use expression_references which indexes into grouping_expressions
        for (auto ref : grouping.expression_references()) {
            if (ref < static_cast<uint32_t>(aggregate.grouping_expressions_size())) {
                auto const& expr = aggregate.grouping_expressions(ref);
                if (expr.has_selection() &&
                    expr.selection().has_direct_reference() &&
                    expr.selection().direct_reference().has_struct_field()) {
                    group_by_columns.push_back(
                        expr.selection().direct_reference().struct_field().field()
                    );
                }
            }
        }
    }

    // Convert aggregate functions
    std::vector<AggregateInfo> aggregates;
    for (auto const& measure : aggregate.measures()) {
        if (measure.has_measure()) {
            auto const& agg_func = measure.measure();
            AggregateInfo info;

            // Get function name
            info.function_name = RemoveExtension(GetFunctionName(agg_func.function_reference()));

            // Get argument column indices
            for (auto const& arg : agg_func.arguments()) {
                if (arg.has_value() &&
                    arg.value().has_selection() &&
                    arg.value().selection().has_direct_reference() &&
                    arg.value().selection().direct_reference().has_struct_field()) {
                    info.argument_indices.push_back(
                        arg.value().selection().direct_reference().struct_field().field()
                    );
                }
            }

            // Get output type
            if (agg_func.has_output_type()) {
                info.output_type = ConvertType(agg_func.output_type());
            }

            // Check if distinct
            info.is_distinct = (agg_func.invocation() ==
                substrait::AggregateFunction::AGGREGATION_INVOCATION_DISTINCT);

            aggregates.push_back(std::move(info));
        }
    }

    return std::make_unique<PhysicalAggregateRel>(
        std::move(child),
        std::move(group_by_columns),
        std::move(aggregates)
    );
}

std::unique_ptr<PhysicalOperator> SubstraitPlanConverter::ConvertSortRel(
    substrait::SortRel const& sort
) {
    // Convert the input relation first
    auto child = ConvertRelation(sort.input());

    // Convert sort fields
    std::vector<SortFieldInfo> sort_fields;
    for (auto const& sf : sort.sorts()) {
        SortFieldInfo info;

        // Get the field index
        if (sf.has_expr() &&
            sf.expr().has_selection() &&
            sf.expr().selection().has_direct_reference() &&
            sf.expr().selection().direct_reference().has_struct_field()) {
            info.field_index = sf.expr().selection().direct_reference().struct_field().field();
        } else {
            throw std::runtime_error("Sort field must be a direct field reference");
        }

        // Get sort direction
        switch (sf.direction()) {
            case substrait::SortField::SORT_DIRECTION_ASC_NULLS_FIRST:
                info.ascending = true;
                info.nulls_first = true;
                break;
            case substrait::SortField::SORT_DIRECTION_ASC_NULLS_LAST:
                info.ascending = true;
                info.nulls_first = false;
                break;
            case substrait::SortField::SORT_DIRECTION_DESC_NULLS_FIRST:
                info.ascending = false;
                info.nulls_first = true;
                break;
            case substrait::SortField::SORT_DIRECTION_DESC_NULLS_LAST:
                info.ascending = false;
                info.nulls_first = false;
                break;
            default:
                info.ascending = true;
                info.nulls_first = true;
        }

        sort_fields.push_back(info);
    }

    return std::make_unique<PhysicalSortRel>(
        std::move(child),
        std::move(sort_fields)
    );
}

std::unique_ptr<PhysicalOperator> SubstraitPlanConverter::ConvertFetchRel(
    substrait::FetchRel const& fetch
) {
    // Convert the input relation first
    auto child = ConvertRelation(fetch.input());

    // Get offset and count
    int64_t offset = 0;
    int64_t count = -1;  // -1 means no limit

    // Handle both old (deprecated) and new offset/count APIs
    switch (fetch.offset_mode_case()) {
        case substrait::FetchRel::kOffset:
            offset = fetch.offset();
            break;
        case substrait::FetchRel::kOffsetExpr:
            // TODO: evaluate expression
            break;
        default:
            break;
    }

    switch (fetch.count_mode_case()) {
        case substrait::FetchRel::kCount:
            count = fetch.count();
            break;
        case substrait::FetchRel::kCountExpr:
            // TODO: evaluate expression
            break;
        default:
            break;
    }

    return std::make_unique<PhysicalFetchRel>(
        std::move(child),
        offset,
        count
    );
}

ExpressionInfo SubstraitPlanConverter::ConvertExpression(
    substrait::Expression const& expr
) {
    ExpressionInfo info;

    switch (expr.rex_type_case()) {
        case substrait::Expression::kLiteral: {
            info.type = ExpressionInfo::Type::LITERAL;
            auto const& lit = expr.literal();

            // Convert literal to string representation
            if (lit.has_boolean()) {
                info.literal_value = lit.boolean() ? "true" : "false";
                info.output_type = cudf::data_type{cudf::type_id::BOOL8};
            } else if (lit.has_i8()) {
                info.literal_value = std::to_string(lit.i8());
                info.output_type = cudf::data_type{cudf::type_id::INT8};
            } else if (lit.has_i16()) {
                info.literal_value = std::to_string(lit.i16());
                info.output_type = cudf::data_type{cudf::type_id::INT16};
            } else if (lit.has_i32()) {
                info.literal_value = std::to_string(lit.i32());
                info.output_type = cudf::data_type{cudf::type_id::INT32};
            } else if (lit.has_i64()) {
                info.literal_value = std::to_string(lit.i64());
                info.output_type = cudf::data_type{cudf::type_id::INT64};
            } else if (lit.has_fp32()) {
                info.literal_value = std::to_string(lit.fp32());
                info.output_type = cudf::data_type{cudf::type_id::FLOAT32};
            } else if (lit.has_fp64()) {
                info.literal_value = std::to_string(lit.fp64());
                info.output_type = cudf::data_type{cudf::type_id::FLOAT64};
            } else if (lit.has_string()) {
                info.literal_value = lit.string();
                info.output_type = cudf::data_type{cudf::type_id::STRING};
            } else if (lit.has_date()) {
                info.literal_value = std::to_string(lit.date());
                info.output_type = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
            }
            break;
        }

        case substrait::Expression::kSelection: {
            info.type = ExpressionInfo::Type::FIELD_REFERENCE;
            auto const& sel = expr.selection();
            if (sel.has_direct_reference() &&
                sel.direct_reference().has_struct_field()) {
                info.field_index = sel.direct_reference().struct_field().field();
            }
            break;
        }

        case substrait::Expression::kScalarFunction: {
            info.type = ExpressionInfo::Type::SCALAR_FUNCTION;
            auto const& func = expr.scalar_function();

            info.function_name = RemoveExtension(GetFunctionName(func.function_reference()));

            // Convert function output type
            if (func.has_output_type()) {
                info.output_type = ConvertType(func.output_type());
            }

            // Convert arguments
            for (auto const& arg : func.arguments()) {
                if (arg.has_value()) {
                    info.arguments.push_back(ConvertExpression(arg.value()));
                }
            }
            break;
        }

        case substrait::Expression::kCast: {
            info.type = ExpressionInfo::Type::CAST;
            auto const& cast = expr.cast();

            if (cast.has_type()) {
                info.output_type = ConvertType(cast.type());
            }
            if (cast.has_input()) {
                info.arguments.push_back(ConvertExpression(cast.input()));
            }
            break;
        }

        case substrait::Expression::kIfThen: {
            info.type = ExpressionInfo::Type::IF_THEN;
            info.function_name = "if_then";
            // TODO: Handle if-then expressions
            break;
        }

        default:
            info.type = ExpressionInfo::Type::UNKNOWN;
            break;
    }

    return info;
}

cudf::data_type SubstraitPlanConverter::ConvertType(substrait::Type const& type) {
    switch (type.kind_case()) {
        case substrait::Type::kBool:
            return cudf::data_type{cudf::type_id::BOOL8};
        case substrait::Type::kI8:
            return cudf::data_type{cudf::type_id::INT8};
        case substrait::Type::kI16:
            return cudf::data_type{cudf::type_id::INT16};
        case substrait::Type::kI32:
            return cudf::data_type{cudf::type_id::INT32};
        case substrait::Type::kI64:
            return cudf::data_type{cudf::type_id::INT64};
        case substrait::Type::kFp32:
            return cudf::data_type{cudf::type_id::FLOAT32};
        case substrait::Type::kFp64:
            return cudf::data_type{cudf::type_id::FLOAT64};
        case substrait::Type::kString:
        case substrait::Type::kVarchar:
            return cudf::data_type{cudf::type_id::STRING};
        case substrait::Type::kDate:
            return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
        case substrait::Type::kDecimal:
            // For now, map decimals to float64
            return cudf::data_type{cudf::type_id::FLOAT64};
        default:
            return cudf::data_type{cudf::type_id::EMPTY};
    }
}

std::vector<cudf::data_type> SubstraitPlanConverter::ConvertSchema(
    substrait::NamedStruct const& schema
) {
    std::vector<cudf::data_type> types;
    if (schema.has_struct_()) {
        for (auto const& field_type : schema.struct_().types()) {
            types.push_back(ConvertType(field_type));
        }
    }
    return types;
}

std::string SubstraitPlanConverter::GetFunctionName(uint32_t anchor) const {
    return parser_.FindFunction(anchor);
}

std::string SubstraitPlanConverter::RemoveExtension(std::string const& name) {
    // Remove extension suffix (e.g., "add:i32_i32" -> "add")
    auto pos = name.find(':');
    if (pos != std::string::npos) {
        return name.substr(0, pos);
    }
    return name;
}

}  // namespace rapidsmpf_substrait

