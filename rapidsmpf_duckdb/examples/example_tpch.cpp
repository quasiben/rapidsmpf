/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Example: Testing rapidsmpf_duckdb extension with TPCH data
 * 
 * This example demonstrates using DuckDB with parquet files.
 * The rapidsmpf_query function is stubbed out for development.
 */

#include <iostream>
#include <chrono>
#include <string>

#include "duckdb.hpp"

void print_separator() {
    std::cout << std::string(70, '=') << std::endl;
}

void run_query(duckdb::Connection& con, const std::string& description, 
               const std::string& sql) {
    print_separator();
    std::cout << description << std::endl;
    print_separator();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto result = con.Query(sql);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    );
    
    if (result->HasError()) {
        std::cerr << "Error: " << result->GetError() << std::endl;
        return;
    }
    
    std::cout << "Result (" << result->RowCount() << " rows):" << std::endl;
    result->Print();
    std::cout << "\nExecution time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::string tpch_path = "/raid/rapidsmpf/data/tpch/scale-1000";
    
    if (argc > 1) {
        tpch_path = argv[1];
    }
    
    std::cout << std::endl;
    print_separator();
    std::cout << "RapidsMPF DuckDB Extension - TPCH Example" << std::endl;
    std::cout << "TPCH Data Path: " << tpch_path << std::endl;
    print_separator();
    std::cout << std::endl;
    
    // Create in-memory database
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    
    // Create views for TPCH tables (using parquet files)
    std::cout << "Creating TPCH table views from parquet files..." << std::endl;
    
    con.Query("CREATE VIEW nation AS SELECT * FROM read_parquet('" + 
              tpch_path + "/nation/*.parquet')");
    con.Query("CREATE VIEW region AS SELECT * FROM read_parquet('" + 
              tpch_path + "/region/*.parquet')");
    con.Query("CREATE VIEW customer AS SELECT * FROM read_parquet('" + 
              tpch_path + "/customer/*.parquet')");
    con.Query("CREATE VIEW orders AS SELECT * FROM read_parquet('" + 
              tpch_path + "/orders/*.parquet')");
    con.Query("CREATE VIEW lineitem AS SELECT * FROM read_parquet('" + 
              tpch_path + "/lineitem/*.parquet')");
    con.Query("CREATE VIEW part AS SELECT * FROM read_parquet('" + 
              tpch_path + "/part/*.parquet')");
    con.Query("CREATE VIEW partsupp AS SELECT * FROM read_parquet('" + 
              tpch_path + "/partsupp/*.parquet')");
    con.Query("CREATE VIEW supplier AS SELECT * FROM read_parquet('" + 
              tpch_path + "/supplier/*.parquet')");
    
    std::cout << "Views created!" << std::endl << std::endl;
    
    // Query 1: Nation and Region (small tables)
    run_query(con, "Query 1: Nations and Regions",
        R"(
        SELECT n.n_name AS nation, r.r_name AS region
        FROM nation n
        JOIN region r ON n.n_regionkey = r.r_regionkey
        ORDER BY region, nation
        )"
    );
    
    // Query 2: TPCH Q1 - Pricing Summary Report
    run_query(con, "Query 2: TPCH Q1 - Pricing Summary (on single partition)",
        R"(
        SELECT
            l_returnflag,
            l_linestatus,
            SUM(l_quantity) AS sum_qty,
            SUM(l_extendedprice) AS sum_base_price,
            SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
            AVG(l_quantity) AS avg_qty,
            AVG(l_extendedprice) AS avg_price,
            AVG(l_discount) AS avg_disc,
            COUNT(*) AS count_order
        FROM read_parquet('/raid/rapidsmpf/data/tpch/scale-1000/lineitem/part.0.parquet')
        WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
        )"
    );
    
    // Query 3: Customer order counts
    run_query(con, "Query 3: Top 10 Customers by Order Count",
        R"(
        SELECT 
            c.c_name,
            c.c_nationkey,
            COUNT(*) AS order_count,
            SUM(o.o_totalprice) AS total_value
        FROM customer c
        JOIN orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey, c.c_name, c.c_nationkey
        ORDER BY total_value DESC
        LIMIT 10
        )"
    );
    
    // Query 4: Revenue by nation (simplified)
    run_query(con, "Query 4: Revenue by Nation (from first lineitem partition)",
        R"(
        SELECT 
            n.n_name AS nation,
            SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
        FROM read_parquet('/raid/rapidsmpf/data/tpch/scale-1000/lineitem/part.0.parquet') l
        JOIN orders o ON l.l_orderkey = o.o_orderkey
        JOIN customer c ON o.o_custkey = c.c_custkey
        JOIN nation n ON c.c_nationkey = n.n_nationkey
        GROUP BY n.n_name
        ORDER BY revenue DESC
        LIMIT 10
        )"
    );
    
    // Performance benchmark
    print_separator();
    std::cout << "Performance Benchmark: TPCH Q1 on Full Lineitem" << std::endl;
    print_separator();
    
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = con.Query(R"(
            SELECT
                l_returnflag,
                l_linestatus,
                SUM(l_quantity) AS sum_qty,
                SUM(l_extendedprice) AS sum_base_price,
                COUNT(*) AS count_order
            FROM lineitem
            WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
            GROUP BY l_returnflag, l_linestatus
            ORDER BY l_returnflag, l_linestatus
        )");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        );
        
        if (!result->HasError()) {
            result->Print();
            std::cout << "\nFull lineitem (scale 1000) query time: " 
                      << duration.count() << " ms" << std::endl;
        } else {
            std::cerr << "Error: " << result->GetError() << std::endl;
        }
    }
    
    std::cout << std::endl;
    print_separator();
    std::cout << "Example completed!" << std::endl;
    print_separator();
    
    return 0;
}

