# rapidsmpf_substrait Architecture

This document describes the architecture of the `rapidsmpf_substrait` library, which provides a [Substrait](https://substrait.io/) interface for GPU-accelerated query execution using rapidsmpf.

## Overview

`rapidsmpf_substrait` enables execution of Substrait query plans on GPU by:
1. Parsing Substrait plans (binary protobuf or JSON format)
2. Converting Substrait relations to rapidsmpf physical operators
3. Building a streaming pipeline of coroutine-based nodes
4. Executing the pipeline on GPU using libcudf

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Substrait Plan (JSON/Binary)                       │
│                                                                              │
│   {                                                                          │
│     "relations": [{                                                          │
│       "root": {                                                              │
│         "input": { "filter": { ... } }                                       │
│       }                                                                      │
│     }]                                                                       │
│   }                                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SubstraitPlanParser                                  │
│                                                                              │
│   - ParseFromJson() / ParseFromBinary() / ParseFromFile()                   │
│   - Builds function map from extension declarations                          │
│   - Provides access to root relation                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       SubstraitPlanConverter                                 │
│                                                                              │
│   Converts Substrait relations to PhysicalOperators:                         │
│   - ReadRel → PhysicalReadRel                                               │
│   - FilterRel → PhysicalFilterRel                                           │
│   - ProjectRel → PhysicalProjectRel                                         │
│   - AggregateRel → PhysicalAggregateRel                                     │
│   - SortRel → PhysicalSortRel                                               │
│   - FetchRel → PhysicalFetchRel                                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Physical Operator Tree                               │
│                                                                              │
│   PhysicalAggregateRel                                                       │
│       └── PhysicalFilterRel                                                  │
│               └── PhysicalReadRel (parquet files)                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Executor                                        │
│                                                                              │
│   - Creates rapidsmpf streaming Context                                      │
│   - Creates Channels between operators                                       │
│   - Calls BuildNode() on each PhysicalOperator                              │
│   - Runs the streaming pipeline                                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       RapidsMPF Streaming (GPU)                              │
│                                                                              │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐             │
│   │  read_parquet │────▶│    filter     │────▶│   aggregate   │──▶ Result   │
│   │   (Source)    │     │  (Transform)  │     │    (Sink)     │             │
│   └───────────────┘     └───────────────┘     └───────────────┘             │
│          │                     │                     │                       │
│          ▼                     ▼                     ▼                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         libcudf (GPU)                               │   │
│   │   - cudf::io::read_parquet()  - cudf::compute_column() (AST)       │   │
│   │   - cudf::reduce()            - cudf::sort_by_key()                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### SubstraitPlanParser

Responsible for parsing Substrait plans from various formats:

```cpp
// Parse from JSON
auto plan = SubstraitPlanParser::ParseFromJson(json_string);

// Parse from binary protobuf
auto plan = SubstraitPlanParser::ParseFromBinary(binary_data);

// Parse from file (auto-detects format)
auto plan = SubstraitPlanParser::ParseFromFile("plan.json");
```

The parser also builds a map of extension function declarations, allowing lookup of function names by anchor ID.

### SubstraitPlanConverter

Recursively converts Substrait relations to a tree of `PhysicalOperator` nodes:

| Substrait Relation | Physical Operator | Description |
|-------------------|-------------------|-------------|
| ReadRel | PhysicalReadRel | Reads parquet files |
| FilterRel | PhysicalFilterRel | Filters rows using cudf AST |
| ProjectRel | PhysicalProjectRel | Selects/computes columns |
| AggregateRel | PhysicalAggregateRel | Aggregates using cudf::reduce |
| SortRel | PhysicalSortRel | Sorts using cudf::sort_by_key |
| FetchRel | PhysicalFetchRel | LIMIT/OFFSET using cudf::slice |

### PhysicalOperator

Base class for all physical operators. Each operator implements:

```cpp
virtual rapidsmpf::streaming::Node BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) = 0;
```

This method creates a C++20 coroutine that processes data according to the operator's semantics.

### Executor

Orchestrates the execution pipeline:

1. Creates the rapidsmpf streaming context
2. Traverses the operator tree (post-order)
3. Creates channels between operators
4. Builds streaming nodes by calling `BuildNode()` on each operator
5. Runs the pipeline using `rapidsmpf::streaming::run_streaming_pipeline()`

## Supported Substrait Features

### Relations

| Relation | Status | Notes |
|----------|--------|-------|
| ReadRel (named_table) | ✅ | Requires table registry |
| ReadRel (local_files) | ✅ | Parquet files |
| FilterRel | ✅ | Numeric comparisons |
| ProjectRel | ✅ | Column selection |
| AggregateRel (ungrouped) | ✅ | SUM, AVG, MIN, MAX, COUNT |
| AggregateRel (grouped) | ❌ | Not yet implemented |
| SortRel | ✅ | Multi-column, ASC/DESC |
| FetchRel | ✅ | LIMIT/OFFSET |
| JoinRel | ❌ | Not yet implemented |
| SetRel | ❌ | Not yet implemented |

### Expressions

| Expression Type | Status | Notes |
|-----------------|--------|-------|
| Literal (numeric) | ✅ | INT8-64, FLOAT32/64 |
| Literal (string) | ✅ | VARCHAR, STRING |
| Literal (date) | ✅ | DATE (days since epoch) |
| FieldReference | ✅ | Direct struct field refs |
| ScalarFunction (comparison) | ✅ | =, !=, <, >, <=, >= |
| ScalarFunction (logical) | ✅ | AND, OR, NOT |
| ScalarFunction (arithmetic) | ❌ | Not yet implemented |
| Cast | ⚠️ | Basic support |
| IfThen | ❌ | Not yet implemented |

### Types

| Substrait Type | cudf Type | Status |
|----------------|-----------|--------|
| bool | BOOL8 | ✅ |
| i8/i16/i32/i64 | INT8-64 | ✅ |
| fp32/fp64 | FLOAT32/64 | ✅ |
| string/varchar | STRING | ✅ |
| date | TIMESTAMP_DAYS | ✅ |
| decimal | FLOAT64 | ⚠️ Approximated |

## Usage Examples

### Execute from JSON

```cpp
#include <rapidsmpf_substrait.hpp>

std::string json = R"({
  "relations": [{
    "root": {
      "input": { "read": { ... } },
      "names": ["col1", "col2"]
    }
  }]
})";

auto result = rapidsmpf_substrait::ExecuteFromJson(json);
std::cout << "Rows: " << result.rows_processed << std::endl;
```

### Execute from File

```cpp
auto result = rapidsmpf_substrait::ExecuteFromFile("plan.substrait");
```

### Manual Pipeline Construction

```cpp
// Parse and convert
auto plan = SubstraitPlanParser::ParseFromFile("plan.json");
SubstraitPlanParser parser(std::move(plan));
SubstraitPlanConverter converter(parser);
auto physical_plan = converter.Convert();

// Execute
Executor executor;
auto result = executor.Execute(std::move(physical_plan));
```

## Building

Prerequisites:
- rapidsmpf (build the sibling cpp directory first)
- cudf
- protobuf

```bash
# Build rapidsmpf first
cd ../cpp
cmake -B build
cmake --build build

# Build rapidsmpf_substrait
cd ../rapidsmpf_substrait
make release
```

## References

- [Substrait Specification](https://substrait.io/spec/specification/)
- [Substrait GitHub](https://github.com/substrait-io/substrait)
- [rapidsmpf Documentation](../docs/)
- [libcudf Documentation](https://docs.rapids.ai/api/libcudf/stable/)

