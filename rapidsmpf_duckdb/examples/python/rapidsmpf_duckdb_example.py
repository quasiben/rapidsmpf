#!/usr/bin/env python3
"""
Example: Using rapidsmpf_duckdb Extension from Python

This example demonstrates how to:
1. Load the rapidsmpf_duckdb DuckDB extension from Python
2. Register parquet tables for GPU execution
3. Run queries using rapidsmpf_query() for GPU-accelerated execution
4. Compare GPU vs CPU performance

Requirements:
- duckdb Python package: pip install duckdb
- The rapidsmpf_duckdb.duckdb_extension file (built from the C++ extension)
- CUDA-capable GPU with appropriate drivers

Usage:
    python rapidsmpf_duckdb_example.py [tpch_data_path]

Example:
    python rapidsmpf_duckdb_example.py /raid/rapidsmpf/data/tpch/scale-1000
"""

import os
import sys
import time
from pathlib import Path

import duckdb


def get_extension_path():
    """Find the rapidsmpf_duckdb extension file."""
    # Look for extension relative to this script
    script_dir = Path(__file__).parent
    
    # Possible locations
    candidates = [
        script_dir.parent / "build" / "release" / "extension" / "rapidsmpf_duckdb" / "rapidsmpf_duckdb.duckdb_extension",
        script_dir.parent.parent / "build" / "release" / "extension" / "rapidsmpf_duckdb" / "rapidsmpf_duckdb.duckdb_extension",
        Path("/datasets/bzaitlen/GitRepos/rapidsmpf-substrait/rapidsmpf_duckdb/build/release/extension/rapidsmpf_duckdb/rapidsmpf_duckdb.duckdb_extension"),
    ]
    
    for path in candidates:
        if path.exists():
            return str(path.resolve())
    
    raise FileNotFoundError(
        "Could not find rapidsmpf_duckdb.duckdb_extension. "
        "Please build the extension first using: cmake --build build/release"
    )


def print_separator(char="=", width=70):
    """Print a separator line."""
    print(char * width)


def print_header(title):
    """Print a section header."""
    print()
    print_separator()
    print(title)
    print_separator()


def run_query(con, sql, description=None):
    """Run a query and print results with timing."""
    if description:
        print(f"\n{description}")
        print("-" * len(description))
    
    start = time.perf_counter()
    result = con.execute(sql).fetchdf()
    elapsed = (time.perf_counter() - start) * 1000
    
    print(result.to_string(max_rows=20))
    print(f"\nExecution time: {elapsed:.2f} ms")
    print(f"Rows returned: {len(result)}")
    
    return result, elapsed


def benchmark_query(con, sql, description, num_runs=3):
    """
    Benchmark a query by running it multiple times.
    
    Returns:
        tuple: (DataFrame result, list of timings)
    """
    print(f"\n{description}")
    print("-" * len(description))
    
    timings = []
    result = None
    
    for i in range(num_runs):
        start = time.perf_counter()
        result = con.execute(sql).fetchdf()
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f} ms")
    
    avg_time = sum(timings) / len(timings)
    print(f"  Average: {avg_time:.2f} ms")
    
    return result, timings


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        tpch_path = sys.argv[1]
    else:
        tpch_path = "/raid/rapidsmpf/data/tpch/scale-1000"
    
    print_header("RapidsMPF DuckDB Extension - Python Example")
    print(f"TPC-H Data Path: {tpch_path}")
    print(f"DuckDB Version: {duckdb.__version__}")
    
    # Verify data path exists
    if not os.path.exists(tpch_path):
        print(f"\nError: TPC-H data path does not exist: {tpch_path}")
        print("Please provide a valid path to TPC-H parquet data.")
        sys.exit(1)
    
    # Find extension path
    try:
        extension_path = get_extension_path()
        print(f"Extension Path: {extension_path}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    # Create connection with settings for loading unsigned extensions
    print_header("Loading Extension")
    
    # Required to load unsigned (custom-built) extensions - must be set at connection time
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    
    # Load the extension
    try:
        con.execute(f"LOAD '{extension_path}'")
        print("✓ Extension loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load extension: {e}")
        sys.exit(1)
    
    # Verify extension functions are available
    print("\nAvailable rapidsmpf functions:")
    result = con.execute("""
        SELECT function_name 
        FROM duckdb_functions() 
        WHERE function_name LIKE 'rapidsmpf%'
        ORDER BY function_name
    """).fetchall()
    
    for row in result:
        print(f"  - {row[0]}")
    
    # Register TPC-H tables
    print_header("Registering TPC-H Tables")
    
    tables = ["nation", "region", "customer", "orders", "lineitem", 
              "part", "partsupp", "supplier"]
    
    for table in tables:
        table_path = os.path.join(tpch_path, table)
        if os.path.exists(table_path):
            con.execute(f"SELECT rapidsmpf_register_table('{table}', '{table_path}')")
            print(f"  ✓ Registered: {table}")
        else:
            print(f"  ✗ Path not found: {table_path}")
    
    # List registered tables
    print("\nRegistered tables:")
    result = con.execute("SELECT * FROM rapidsmpf_list_tables()").fetchdf()
    print(result.to_string())
    
    # Create schema tables in DuckDB for planning (empty tables with correct schema)
    print_header("Creating Schema Tables for Query Planning")
    
    for table in tables:
        parquet_path = os.path.join(tpch_path, table, "*.parquet")
        # Create table with schema from parquet, limit 0 to avoid loading data
        try:
            con.execute(f"""
                CREATE TABLE {table} AS 
                SELECT * FROM read_parquet('{parquet_path}') LIMIT 0
            """)
            print(f"  ✓ Schema created: {table}")
        except Exception as e:
            print(f"  ✗ Failed to create schema for {table}: {e}")
    
    # Test CPU queries first
    print_header("CPU Query Tests (Standard DuckDB)")
    
    # Simple query on small table
    run_query(con, 
        "SELECT * FROM nation LIMIT 5",
        "Query 1: Nation table (CPU)")
    
    # Aggregation query
    run_query(con,
        "SELECT COUNT(*) as cnt FROM region",
        "Query 2: Region count (CPU)")
    
    # Test GPU queries using rapidsmpf_query
    print_header("GPU Query Tests (RapidsMPF)")
    
    # Note: Results may show 0 rows until the result conversion is fully implemented
    run_query(con,
        "SELECT * FROM rapidsmpf_query('SELECT * FROM nation')",
        "Query 3: Nation table (GPU via rapidsmpf)")
    
    run_query(con,
        "SELECT * FROM rapidsmpf_query('SELECT COUNT(*) as cnt FROM region')",
        "Query 4: Region count (GPU via rapidsmpf)")
    
    # Benchmark comparison
    print_header("Performance Comparison: CPU vs GPU")
    
    benchmark_sql = "SELECT AVG(l_extendedprice) FROM lineitem WHERE l_quantity > 25"
    
    print("\nBenchmark Query:")
    print(f"  {benchmark_sql}")
    
    # CPU benchmark
    print("\n[CPU - Standard DuckDB]")
    cpu_result, cpu_timings = benchmark_query(
        con, 
        benchmark_sql,
        "CPU Execution",
        num_runs=3
    )
    print(f"\nCPU Result:")
    print(cpu_result.to_string())
    
    # GPU benchmark
    print("\n[GPU - RapidsMPF]")
    gpu_sql = f"SELECT * FROM rapidsmpf_query('{benchmark_sql}')"
    gpu_result, gpu_timings = benchmark_query(
        con,
        gpu_sql,
        "GPU Execution",
        num_runs=3
    )
    print(f"\nGPU Result:")
    print(gpu_result.to_string())
    
    # Summary
    print_header("Summary")
    
    cpu_avg = sum(cpu_timings) / len(cpu_timings)
    gpu_avg = sum(gpu_timings) / len(gpu_timings)
    
    print(f"CPU Average Time: {cpu_avg:.2f} ms")
    print(f"GPU Average Time: {gpu_avg:.2f} ms")
    
    if gpu_avg > 0 and cpu_avg > 0:
        if gpu_avg < cpu_avg:
            speedup = cpu_avg / gpu_avg
            print(f"GPU Speedup: {speedup:.2f}x faster")
        else:
            slowdown = gpu_avg / cpu_avg
            print(f"GPU Slowdown: {slowdown:.2f}x slower (may need warmup or larger data)")
    
    # Clean up
    print_header("Cleanup")
    con.execute("SELECT rapidsmpf_clear_tables()")
    print("✓ Table registry cleared")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

