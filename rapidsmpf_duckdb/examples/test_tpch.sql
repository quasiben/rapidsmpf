-- =============================================================================
-- SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
-- SPDX-License-Identifier: Apache-2.0
-- =============================================================================
-- 
-- Test script for rapidsmpf_duckdb extension with TPCH data
-- 

-- First, let's try a simple query using standard DuckDB to verify data access
SELECT 'Testing standard DuckDB parquet reading...' AS status;

-- Read nation table (small table for quick test)
SELECT COUNT(*) AS nation_count 
FROM read_parquet('/raid/rapidsmpf/data/tpch/scale-1000/nation/*.parquet');

-- Show first few rows
SELECT * FROM read_parquet('/raid/rapidsmpf/data/tpch/scale-1000/nation/*.parquet') LIMIT 5;

-- Read region table
SELECT 'Region table:' AS status;
SELECT * FROM read_parquet('/raid/rapidsmpf/data/tpch/scale-1000/region/*.parquet');

-- Try a simple aggregation on lineitem (larger table)
SELECT 'Lineitem aggregation (first 1M rows):' AS status;
SELECT 
    l_returnflag,
    l_linestatus,
    COUNT(*) AS count_order,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price
FROM read_parquet('/raid/rapidsmpf/data/tpch/scale-1000/lineitem/part.0.parquet')
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;

-- Now try the rapidsmpf_query function
SELECT 'Testing rapidsmpf_query function...' AS status;

-- Simple test of the extension
SELECT * FROM rapidsmpf_query('SELECT 1 as test');

