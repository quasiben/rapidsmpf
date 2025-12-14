/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/rapidsmpf_duckdb_extension.hpp"

#include "duckdb/main/extension_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/planner/binder.hpp"

#include "include/physical_plan_generator.hpp"
#include "include/executor.hpp"
#include "include/table_registry.hpp"

namespace duckdb {

// ============================================================================
// Table function bind data
// ============================================================================

struct RapidsmpfQueryBindData : public TableFunctionData {
    std::string sql_query;
    std::vector<LogicalType> return_types;
    std::vector<std::string> return_names;
    std::unique_ptr<rapidsmpf_duckdb::PhysicalOperator> plan;
};

// ============================================================================
// Table function implementation
// ============================================================================

/**
 * @brief Bind function for rapidsmpf_query.
 * 
 * Parses the SQL, extracts the logical plan, and converts to physical plan.
 */
static unique_ptr<FunctionData> RapidsmpfQueryBind(
    ClientContext& context,
    TableFunctionBindInput& input,
    vector<LogicalType>& return_types,
    vector<string>& names
) {
    auto result = make_uniq<RapidsmpfQueryBindData>();
    
    // Get the SQL query from the first argument
    result->sql_query = input.inputs[0].GetValue<string>();
    
    // Parse the SQL query
    Parser parser;
    parser.ParseQuery(result->sql_query);
    if (parser.statements.empty()) {
        throw InvalidInputException("Empty SQL query");
    }
    if (parser.statements.size() > 1) {
        throw InvalidInputException("Only single statements are supported");
    }
    
    // Get the logical plan using the provided context
    Planner planner(context);
    try {
        planner.CreatePlan(std::move(parser.statements[0]));
    } catch (std::exception& e) {
        throw InvalidInputException("Failed to plan query: " + std::string(e.what()));
    }
    
    auto logical_plan = std::move(planner.plan);
    
    if (!logical_plan) {
        throw InvalidInputException("Query planning returned null plan");
    }
    
    // Important: Resolve the types on the logical plan
    // This populates the types vector which is initially empty
    logical_plan->ResolveOperatorTypes();
    
    // Get the column names from the planner's names vector
    auto& plan_names = planner.names;
    
    // Extract the return types from the logical plan
    for (auto& type : logical_plan->types) {
        return_types.push_back(type);
        result->return_types.push_back(type);
    }
    
    // Generate column names from the planner if available
    for (idx_t i = 0; i < return_types.size(); i++) {
        std::string col_name;
        if (i < plan_names.size()) {
            col_name = plan_names[i];
        } else {
            col_name = "column" + std::to_string(i);
        }
        names.push_back(col_name);
        result->return_names.push_back(col_name);
    }
    
    // If still no columns, this is an error - provide helpful debugging info
    if (return_types.empty()) {
        throw InvalidInputException(
            "Query produced no output columns. This may happen if the query references " 
            "tables that don't exist in the current database context. Query: " + 
            result->sql_query + " (Logical plan type: " + 
            std::to_string(static_cast<int>(logical_plan->type)) + ")"
        );
    }
    
    // Convert to rapidsmpf physical plan
    try {
        rapidsmpf_duckdb::PhysicalPlanGenerator planner_gen(context);
        result->plan = planner_gen.CreatePlan(std::move(logical_plan));
    } catch (std::exception& e) {
        throw InvalidInputException(
            "Failed to create rapidsmpf physical plan: " + std::string(e.what()) +
            ". Note: rapidsmpf_query currently only supports simple queries without " 
            "JOIN, ORDER BY, LIMIT, DISTINCT, or GROUP BY."
        );
    }
    
    return result;
}

/**
 * @brief State for scanning rapidsmpf query results.
 */
struct RapidsmpfQueryState : public GlobalTableFunctionState {
    std::vector<DataChunk> results;
    idx_t current_chunk = 0;
    bool executed = false;
    
    idx_t MaxThreads() const override {
        return 1;  // Single-threaded for now
    }
};

/**
 * @brief Initialize global state for rapidsmpf_query.
 */
static unique_ptr<GlobalTableFunctionState> RapidsmpfQueryInit(
    ClientContext& context,
    TableFunctionInitInput& input
) {
    return make_uniq<RapidsmpfQueryState>();
}

/**
 * @brief Execute the rapidsmpf query and return results.
 */
static void RapidsmpfQueryFunction(
    ClientContext& context,
    TableFunctionInput& data,
    DataChunk& output
) {
    auto& bind_data = data.bind_data->CastNoConst<RapidsmpfQueryBindData>();
    auto& state = data.global_state->Cast<RapidsmpfQueryState>();
    
    // Execute the query if not already done
    if (!state.executed) {
        rapidsmpf_duckdb::Executor executor;
        state.results = executor.Execute(std::move(bind_data.plan));
        state.executed = true;
    }
    
    // Return the next chunk
    if (state.current_chunk >= state.results.size()) {
        output.SetCardinality(0);
        return;
    }
    
    // Copy the chunk to output
    output.Reference(state.results[state.current_chunk]);
    state.current_chunk++;
}

// ============================================================================
// Table registration functions
// ============================================================================

/**
 * @brief Register a table with parquet file path(s).
 *
 * SQL usage:
 *   SELECT rapidsmpf_register_table('lineitem', '/path/to/lineitem/');
 *   SELECT rapidsmpf_register_table('nation', '/path/to/nation.parquet');
 *
 * @param table_name Name of the table for SQL queries
 * @param path Path to parquet file or directory containing parquet files
 */
static void RapidsmpfRegisterTableFunction(
    DataChunk& args,
    ExpressionState& state,
    Vector& result
) {
    auto table_names_data = FlatVector::GetData<string_t>(args.data[0]);
    auto paths_data = FlatVector::GetData<string_t>(args.data[1]);
    auto result_data = FlatVector::GetData<string_t>(result);
    
    for (idx_t i = 0; i < args.size(); i++) {
        auto table_name = table_names_data[i].GetString();
        auto path = paths_data[i].GetString();
        
        try {
            rapidsmpf_duckdb::TableRegistry::Instance().RegisterTable(
                table_name, path
            );
            result_data[i] = StringVector::AddString(
                result, "Registered table '" + table_name + "' with path: " + path
            );
        } catch (std::exception& e) {
            throw InvalidInputException(
                "Failed to register table: " + std::string(e.what())
            );
        }
    }
}

/**
 * @brief Unregister a table.
 *
 * SQL usage:
 *   SELECT rapidsmpf_unregister_table('lineitem');
 */
static void RapidsmpfUnregisterTableFunction(
    DataChunk& args,
    ExpressionState& state,
    Vector& result
) {
    auto table_names_data = FlatVector::GetData<string_t>(args.data[0]);
    auto result_data = FlatVector::GetData<string_t>(result);
    
    for (idx_t i = 0; i < args.size(); i++) {
        auto table_name = table_names_data[i].GetString();
        bool removed = rapidsmpf_duckdb::TableRegistry::Instance()
            .UnregisterTable(table_name);
        if (removed) {
            result_data[i] = StringVector::AddString(
                result, "Unregistered table: " + table_name
            );
        } else {
            result_data[i] = StringVector::AddString(
                result, "Table not found: " + table_name
            );
        }
    }
}

/**
 * @brief List all registered tables.
 */
struct ListTablesBindData : public TableFunctionData {
    std::vector<std::string> table_names;
};

static unique_ptr<FunctionData> RapidsmpfListTablesBind(
    ClientContext& context,
    TableFunctionBindInput& input,
    vector<LogicalType>& return_types,
    vector<string>& names
) {
    auto result = make_uniq<ListTablesBindData>();
    result->table_names = rapidsmpf_duckdb::TableRegistry::Instance().GetTableNames();
    
    return_types.push_back(LogicalType::VARCHAR);
    names.push_back("table_name");
    
    return_types.push_back(LogicalType::VARCHAR);
    names.push_back("file_paths");
    
    return result;
}

struct ListTablesState : public GlobalTableFunctionState {
    idx_t current_row = 0;
};

static unique_ptr<GlobalTableFunctionState> RapidsmpfListTablesInit(
    ClientContext& context,
    TableFunctionInitInput& input
) {
    return make_uniq<ListTablesState>();
}

static void RapidsmpfListTablesFunction(
    ClientContext& context,
    TableFunctionInput& data,
    DataChunk& output
) {
    auto& bind_data = data.bind_data->Cast<ListTablesBindData>();
    auto& state = data.global_state->Cast<ListTablesState>();
    
    idx_t count = 0;
    auto& registry = rapidsmpf_duckdb::TableRegistry::Instance();
    
    while (state.current_row < bind_data.table_names.size() && count < STANDARD_VECTOR_SIZE) {
        auto& table_name = bind_data.table_names[state.current_row];
        auto table_info = registry.GetTable(table_name);
        
        if (table_info) {
            // Table name
            FlatVector::GetData<string_t>(output.data[0])[count] = 
                StringVector::AddString(output.data[0], table_name);
            
            // File paths (as comma-separated list)
            std::string paths_str;
            for (size_t i = 0; i < table_info->file_paths.size(); ++i) {
                if (i > 0) paths_str += ", ";
                paths_str += table_info->file_paths[i];
            }
            FlatVector::GetData<string_t>(output.data[1])[count] = 
                StringVector::AddString(output.data[1], paths_str);
            
            count++;
        }
        state.current_row++;
    }
    
    output.SetCardinality(count);
}

/**
 * @brief Clear all registered tables.
 */
static void RapidsmpfClearTablesFunction(
    DataChunk& args,
    ExpressionState& state,
    Vector& result
) {
    rapidsmpf_duckdb::TableRegistry::Instance().Clear();
    result.SetValue(0, Value("All tables cleared"));
}

// ============================================================================
// Extension registration
// ============================================================================

void RapidsmpfDuckdbExtension::Load(DuckDB& db) {
    // Create the rapidsmpf_query table function
    TableFunction rapidsmpf_query_func(
        "rapidsmpf_query",
        {LogicalType::VARCHAR},  // SQL query string
        RapidsmpfQueryFunction,
        RapidsmpfQueryBind,
        RapidsmpfQueryInit
    );
    ExtensionUtil::RegisterFunction(*db.instance, rapidsmpf_query_func);
    
    // Register table management functions
    
    // rapidsmpf_register_table(table_name, path) -> status message
    ScalarFunction register_func(
        "rapidsmpf_register_table",
        {LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::VARCHAR,
        RapidsmpfRegisterTableFunction
    );
    ExtensionUtil::RegisterFunction(*db.instance, register_func);
    
    // rapidsmpf_unregister_table(table_name) -> status message
    ScalarFunction unregister_func(
        "rapidsmpf_unregister_table",
        {LogicalType::VARCHAR},
        LogicalType::VARCHAR,
        RapidsmpfUnregisterTableFunction
    );
    ExtensionUtil::RegisterFunction(*db.instance, unregister_func);
    
    // rapidsmpf_list_tables() -> table of (table_name, file_paths)
    TableFunction list_func(
        "rapidsmpf_list_tables",
        {},
        RapidsmpfListTablesFunction,
        RapidsmpfListTablesBind,
        RapidsmpfListTablesInit
    );
    ExtensionUtil::RegisterFunction(*db.instance, list_func);
    
    // rapidsmpf_clear_tables() -> status message
    ScalarFunction clear_func(
        "rapidsmpf_clear_tables",
        {},
        LogicalType::VARCHAR,
        RapidsmpfClearTablesFunction
    );
    ExtensionUtil::RegisterFunction(*db.instance, clear_func);
}

std::string RapidsmpfDuckdbExtension::Name() {
    return "rapidsmpf_duckdb";
}

}  // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void rapidsmpf_duckdb_init(duckdb::DatabaseInstance& db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::RapidsmpfDuckdbExtension>();
}

DUCKDB_EXTENSION_API const char* rapidsmpf_duckdb_version() {
    return duckdb::DuckDB::LibraryVersion();
}

}
