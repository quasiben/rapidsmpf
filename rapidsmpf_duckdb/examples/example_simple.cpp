/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Simple Example: DuckDB with in-memory tables
 * 
 * This example creates toy data in-memory and runs queries.
 * No external files needed.
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

int main() {
    std::cout << std::endl;
    print_separator();
    std::cout << "RapidsMPF DuckDB Extension - Simple In-Memory Example" << std::endl;
    print_separator();
    std::cout << std::endl;
    
    // Create in-memory database
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    
    // Create sample tables
    std::cout << "Creating sample tables..." << std::endl;
    
    // Create a simple orders table with 100K rows
    con.Query(R"(
        CREATE TABLE orders AS
        SELECT
            i AS order_id,
            (i % 1000) + 1 AS customer_id,
            (i % 100) + 1 AS product_id,
            (random() * 9 + 1)::INT AS quantity,
            (random() * 990 + 10)::DECIMAL(10,2) AS price,
            DATE '2024-01-01' + INTERVAL (i % 365) DAY AS order_date
        FROM generate_series(1, 100000) t(i)
    )");
    std::cout << "  - orders: 100,000 rows" << std::endl;
    
    // Create customers table
    con.Query(R"(
        CREATE TABLE customers AS
        SELECT
            i AS customer_id,
            'Customer_' || i AS name,
            CASE (i % 4)
                WHEN 0 THEN 'North'
                WHEN 1 THEN 'South'
                WHEN 2 THEN 'East'
                ELSE 'West'
            END AS region
        FROM generate_series(1, 1000) t(i)
    )");
    std::cout << "  - customers: 1,000 rows" << std::endl;
    
    // Create products table
    con.Query(R"(
        CREATE TABLE products AS
        SELECT
            i AS product_id,
            'Product_' || i AS product_name,
            CASE (i % 4)
                WHEN 0 THEN 'Electronics'
                WHEN 1 THEN 'Clothing'
                WHEN 2 THEN 'Food'
                ELSE 'Books'
            END AS category,
            (random() * 100 + 10)::DECIMAL(10,2) AS unit_cost
        FROM generate_series(1, 100) t(i)
    )");
    std::cout << "  - products: 100 rows" << std::endl;
    
    std::cout << "\nTables created!" << std::endl << std::endl;
    
    // Query 1: Simple SELECT
    run_query(con, "Query 1: Simple SELECT with LIMIT",
        "SELECT * FROM orders LIMIT 10"
    );
    
    // Query 2: Filter (WHERE)
    run_query(con, "Query 2: Filter (WHERE clause)",
        R"(
        SELECT order_id, customer_id, price
        FROM orders
        WHERE price > 800
        LIMIT 10
        )"
    );
    
    // Query 3: Aggregation
    run_query(con, "Query 3: Simple Aggregation",
        R"(
        SELECT
            COUNT(*) AS total_orders,
            SUM(price) AS total_revenue,
            AVG(price) AS avg_price,
            MIN(price) AS min_price,
            MAX(price) AS max_price
        FROM orders
        )"
    );
    
    // Query 4: GROUP BY aggregation
    run_query(con, "Query 4: GROUP BY Aggregation",
        R"(
        SELECT
            customer_id,
            COUNT(*) AS order_count,
            SUM(price) AS total_spent
        FROM orders
        GROUP BY customer_id
        ORDER BY total_spent DESC
        LIMIT 10
        )"
    );
    
    // Query 5: JOIN
    run_query(con, "Query 5: JOIN Query",
        R"(
        SELECT
            c.region,
            COUNT(*) AS order_count,
            SUM(o.price) AS total_revenue
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY c.region
        ORDER BY total_revenue DESC
        )"
    );
    
    // Query 6: Multi-table JOIN with filter
    run_query(con, "Query 6: Multi-table JOIN with Filter",
        R"(
        SELECT
            p.category,
            c.region,
            COUNT(*) AS num_orders,
            SUM(o.price * o.quantity) AS revenue
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN products p ON o.product_id = p.product_id
        WHERE o.price > 100
        GROUP BY p.category, c.region
        ORDER BY revenue DESC
        LIMIT 10
        )"
    );
    
    // Query 7: Subquery
    run_query(con, "Query 7: Subquery",
        R"(
        SELECT customer_id, total_spent
        FROM (
            SELECT
                customer_id,
                SUM(price) AS total_spent
            FROM orders
            GROUP BY customer_id
        ) sub
        WHERE total_spent > 5000
        ORDER BY total_spent DESC
        LIMIT 10
        )"
    );
    
    // Query 8: Window function
    run_query(con, "Query 8: Window Function",
        R"(
        SELECT
            order_id,
            customer_id,
            price,
            SUM(price) OVER (PARTITION BY customer_id ORDER BY order_id) AS running_total
        FROM orders
        WHERE customer_id <= 5
        ORDER BY customer_id, order_id
        LIMIT 20
        )"
    );
    
    // Performance benchmark
    print_separator();
    std::cout << "Performance Benchmark" << std::endl;
    print_separator();
    
    // Run aggregation multiple times
    const int num_runs = 5;
    long total_time = 0;
    
    for (int i = 0; i < num_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = con.Query(R"(
            SELECT
                customer_id,
                COUNT(*) AS cnt,
                SUM(price) AS total,
                AVG(quantity) AS avg_qty
            FROM orders
            GROUP BY customer_id
        )");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start
        );
        total_time += duration.count();
        
        std::cout << "Run " << (i + 1) << ": " << duration.count() / 1000.0 
                  << " ms (" << result->RowCount() << " rows)" << std::endl;
    }
    
    std::cout << "\nAverage time: " << (total_time / num_runs) / 1000.0 
              << " ms" << std::endl;
    
    std::cout << std::endl;
    print_separator();
    std::cout << "Example completed successfully!" << std::endl;
    print_separator();
    
    return 0;
}

