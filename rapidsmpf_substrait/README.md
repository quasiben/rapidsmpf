# rapidsmpf_substrait

A [Substrait](https://substrait.io/) interface for GPU-accelerated query execution using [rapidsmpf](../cpp/).

## Overview

`rapidsmpf_substrait` enables execution of Substrait query plans on GPU by:

1. **Parsing** Substrait plans (binary protobuf or JSON format)
2. **Converting** Substrait relations to rapidsmpf physical operators
3. **Executing** on GPU using libcudf via the rapidsmpf streaming framework

## Quick Start

```cpp
#include <rapidsmpf_substrait.hpp>

// Execute from JSON
auto result = rapidsmpf_substrait::ExecuteFromJson(json_plan);
std::cout << "Rows: " << result.rows_processed << std::endl;
std::cout << "Time: " << result.execution_time_ms << " ms" << std::endl;

// Or from file
auto result = rapidsmpf_substrait::ExecuteFromFile("plan.substrait");
```

## Building

### Prerequisites

1. Build rapidsmpf first:
```bash
cd ../cpp
cmake -B build
cmake --build build
```

2. Ensure protobuf is installed:
```bash
# Ubuntu/Debian
apt install libprotobuf-dev protobuf-compiler

# Conda
conda install protobuf libprotobuf
```

### Build

```bash
# Activate conda environment (if using)
conda activate 2025-12-12-rapidsmpf-duckdb

# Build release
make release

# Or debug
make debug
```

### Run Examples

```bash
# Basic example
./build/release/example_substrait

# JSON parsing example
./build/release/example_from_json

# Execute a plan file
./build/release/example_substrait my_plan.json
```

## Supported Operations

| Substrait Relation | Status | Notes |
|-------------------|--------|-------|
| ReadRel (parquet) | ✅ | Local files |
| FilterRel | ✅ | Numeric comparisons |
| ProjectRel | ✅ | Column selection |
| AggregateRel | ✅ | Ungrouped (SUM, AVG, MIN, MAX, COUNT) |
| SortRel | ✅ | Multi-column sorting |
| FetchRel | ✅ | LIMIT/OFFSET |
| JoinRel | ❌ | Not yet implemented |

## Architecture

```
Substrait Plan (JSON/Binary)
         │
         ▼
┌─────────────────────┐
│ SubstraitPlanParser │  ← Parse protobuf
└─────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ SubstraitPlanConverter   │  ← Convert to physical ops
└──────────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Physical Operator   │  ← Build streaming nodes
│ Tree                │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Executor            │  ← Run streaming pipeline
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ GPU Execution       │  ← libcudf operations
│ (rapidsmpf + cudf)  │
└─────────────────────┘
```

## Project Structure

```
rapidsmpf_substrait/
├── CMakeLists.txt
├── Makefile
├── README.md
├── docs/
│   └── architecture.md
├── examples/
│   ├── example_substrait.cpp
│   └── example_from_json.cpp
└── src/
    ├── include/
    │   ├── rapidsmpf_substrait.hpp    # Main header
    │   ├── executor.hpp
    │   ├── physical_operator.hpp
    │   ├── substrait_plan_parser.hpp
    │   ├── substrait_plan_converter.hpp
    │   └── operators/
    │       ├── physical_read_rel.hpp
    │       ├── physical_filter_rel.hpp
    │       ├── physical_project_rel.hpp
    │       ├── physical_aggregate_rel.hpp
    │       ├── physical_sort_rel.hpp
    │       └── physical_fetch_rel.hpp
    ├── executor.cpp
    ├── physical_operator.cpp
    ├── substrait_plan_parser.cpp
    ├── substrait_plan_converter.cpp
    └── operators/
        ├── physical_read_rel.cpp
        ├── physical_filter_rel.cpp
        ├── physical_project_rel.cpp
        ├── physical_aggregate_rel.cpp
        ├── physical_sort_rel.cpp
        └── physical_fetch_rel.cpp
```

## Comparison with rapidsmpf_duckdb

| Feature | rapidsmpf_duckdb | rapidsmpf_substrait |
|---------|------------------|---------------------|
| Input | SQL query string | Substrait plan (JSON/binary) |
| Parser | DuckDB SQL parser | Protobuf parser |
| Intermediate | DuckDB LogicalOperator | Substrait Rel |
| Execution | rapidsmpf streaming | rapidsmpf streaming |
| GPU Backend | libcudf | libcudf |

Both libraries share the same execution model but differ in how they receive query plans. `rapidsmpf_duckdb` uses DuckDB for SQL parsing, while `rapidsmpf_substrait` accepts pre-built Substrait plans from any producer.

## Related Projects

- [Substrait](https://github.com/substrait-io/substrait) - Cross-language query plan specification
- [DuckDB Substrait Extension](../sirius/substrait/) - DuckDB ↔ Substrait conversion
- [Spark Substrait Gateway](https://github.com/voltrondata/spark-substrait-gateway) - Spark → Substrait

## License

Apache-2.0

