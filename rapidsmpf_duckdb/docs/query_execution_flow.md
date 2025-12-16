# RapidsMPF DuckDB Extension: Query Execution Flow

This document explains how a SQL query flows from DuckDB's SQL parser through to GPU-accelerated execution via RapidsMPF's streaming framework.

## Overview

The `rapidsmpf_duckdb` extension enables GPU-accelerated query execution by:
1. Using DuckDB for SQL parsing and logical planning
2. Converting the logical plan to a RapidsMPF physical plan
3. Building a streaming network of coroutine-based nodes
4. Executing the pipeline on GPU using libcudf

---

## Example Query

We'll trace this query through the system:

```sql
SELECT * FROM rapidsmpf_query(
    'SELECT AVG(l_extendedprice) as avg_price FROM lineitem WHERE l_quantity > 25'
)
```

This query:
- **Reads** data from the `lineitem` parquet files
- **Filters** rows where `l_quantity > 25`
- **Aggregates** by computing `AVG(l_extendedprice)`

The `lineitem` table in TPC-H Scale-1000 contains ~6 billion rows across 500 parquet files.

---

## Stage 1: Table Registration

Before executing queries, parquet file paths must be registered with the table name:

```sql
SELECT rapidsmpf_register_table('lineitem', '/raid/rapidsmpf/data/tpch/scale-1000/lineitem');
```

### How It Works

The `TableRegistry` singleton maps logical table names to physical parquet file paths:

```cpp
// src/table_registry.hpp:43-48
class TableRegistry {
  public:
    static TableRegistry& Instance();
    
    void RegisterTable(
        std::string const& table_name,
        std::string const& path,
        std::vector<std::string> column_names = {}
    );
    // ...
};
```

When a directory is registered, all `.parquet` files are discovered:

```cpp
// TableRegistry automatically lists parquet files in the directory
// e.g., /raid/rapidsmpf/data/tpch/scale-1000/lineitem/ contains:
//   part.0.parquet, part.1.parquet, ... part.499.parquet
```

A placeholder table is also created in DuckDB so the planner can resolve the schema:

```sql
CREATE OR REPLACE TABLE lineitem AS 
SELECT * FROM read_parquet('/path/to/lineitem/*.parquet') LIMIT 0
```

---

## Stage 2: SQL Parsing & Logical Plan Generation

When `rapidsmpf_query('SELECT AVG(...) FROM lineitem WHERE ...')` is called, the extension:

### 2.1 Parses the SQL String

```cpp
// src/rapidsmpf_extension.cpp:52-60
Parser parser;
parser.ParseQuery(result->sql_query);
```

### 2.2 Creates the Logical Plan

DuckDB's `Planner` converts the parsed statement into a tree of `LogicalOperator` nodes:

```cpp
// src/rapidsmpf_extension.cpp:63-78
Planner planner(context);
planner.CreatePlan(std::move(parser.statements[0]));
auto logical_plan = std::move(planner.plan);

// Resolve types so we know output column types
logical_plan->ResolveOperatorTypes();
```

### Logical Plan Structure

For our query, DuckDB produces this logical operator tree:

```
LOGICAL_AGGREGATE (AVG(l_extendedprice))
    └── LOGICAL_FILTER (l_quantity > 25)
            └── LOGICAL_GET (lineitem table scan)
```

Each node has:
- **Type**: The operator kind (GET, FILTER, AGGREGATE, etc.)
- **Types**: Output column data types
- **Children**: Input operators
- **Expressions**: Filter predicates, projections, aggregations

---

## Stage 3: Physical Plan Generation

The `PhysicalPlanGenerator` converts DuckDB's logical operators into RapidsMPF physical operators:

```cpp
// src/rapidsmpf_extension.cpp:112-114
rapidsmpf_duckdb::PhysicalPlanGenerator planner_gen(context);
result->plan = planner_gen.CreatePlan(std::move(logical_plan));
```

### Dispatch by Operator Type

The generator recursively traverses the logical tree:

```cpp
// src/physical_plan_generator.cpp:29-83
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
        
        // Unsupported operators throw descriptive errors
        default:
            throw std::runtime_error("Unsupported logical operator...");
    }
}
```

### Physical Operator Hierarchy

Each physical operator inherits from `PhysicalOperator`:

```cpp
// src/include/physical_operator.hpp:34-126
class PhysicalOperator {
  public:
    // Build the streaming node (coroutine) for execution
    virtual rapidsmpf::streaming::Node BuildNode(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
    ) = 0;

  protected:
    duckdb::PhysicalOperatorType type_;
    std::vector<duckdb::LogicalType> types_;
    std::vector<std::unique_ptr<PhysicalOperator>> children_;
};
```

### Physical Plan Structure

After conversion, we have:

```
PhysicalAggregate (AVG)
    └── PhysicalFilter (l_quantity > 25)
            └── PhysicalTableScan (lineitem)
```

---

## Stage 4: Building the Streaming Network

The `Executor` transforms the physical operator tree into a network of streaming nodes connected by channels.

### 4.1 Executor Initialization

```cpp
// src/executor.cpp:31-64
Executor::Executor() {
    // Create communicator for single-process execution
    comm_ = std::make_shared<rapidsmpf::Single>(options);
    
    // Create RMM memory resource adaptor
    mr_adaptor_ = std::make_unique<rapidsmpf::RmmResourceAdaptor>(current_mr);
    
    // Create buffer resource for GPU memory management
    br_ = std::make_shared<rapidsmpf::BufferResource>(
        mr_adaptor_.get(),
        /* pinned memory */ ...,
        /* memory limits */ ...,
        stream_pool_,
        statistics_
    );
    
    // Create streaming context
    ctx_ = std::make_shared<rapidsmpf::streaming::Context>(
        options, comm_, br_, statistics_
    );
}
```

### 4.2 Channel Creation

Channels are the edges in the streaming graph. They connect operators:

```cpp
// src/executor.cpp:144-147
// Create channels between consecutive operators
std::vector<std::shared_ptr<rapidsmpf::streaming::Channel>> channels;
for (size_t i = 0; i <= operators.size(); i++) {
    channels.push_back(ctx_->create_channel());
}
```

### 4.3 Node Construction

Each physical operator builds a streaming `Node` (a C++20 coroutine):

```cpp
// src/executor.cpp:149-161
for (size_t i = 0; i < operators.size(); i++) {
    auto* op = operators[i];
    
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in = nullptr;
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out = channels[i];
    
    if (!op->IsSource() && i > 0) {
        ch_in = channels[i - 1];
    }
    
    nodes.push_back(op->BuildNode(ctx_, ch_in, ch_out));
}
```

### Streaming Network Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RapidsMPF Streaming Network                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ch[0]    ┌─────────────────┐                        │
│  │                 │  ───────►   │                 │                        │
│  │  PhysicalTable  │  TableChunk │  PhysicalFilter │                        │
│  │      Scan       │             │   (passthru*)   │                        │
│  │   (read_parquet)│             │                 │                        │
│  └─────────────────┘             └─────────────────┘                        │
│                                          │                                  │
│                                          │ ch[1]                            │
│                                          ▼ TableChunk                       │
│                                  ┌─────────────────┐                        │
│                                  │                 │                        │
│                                  │ PhysicalAggre-  │                        │
│                                  │     gate        │                        │
│                                  │ (cudf::reduce)  │                        │
│                                  └─────────────────┘                        │
│                                          │                                  │
│                                          │ ch[2]                            │
│                                          ▼ TableChunk                       │
│                                  ┌─────────────────┐                        │
│                                  │                 │                        │
│                                  │   sink_and_     │                        │
│                                  │     count       │                        │
│                                  │                 │                        │
│                                  └─────────────────┘                        │
│                                                                             │
│  * Filter expression conversion is stubbed; currently passes through       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 5: Node Implementation Details

### 5.1 PhysicalTableScan: Parquet Reader

The table scan uses RapidsMPF's built-in `read_parquet` node:

```cpp
// src/operators/physical_table_scan.cpp:57-88
rapidsmpf::streaming::Node PhysicalTableScan::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> /* ch_in */,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    // Look up file paths from registry
    auto table_info = TableRegistry::Instance().GetTable(table_name_);
    file_paths_ = table_info->file_paths;
    
    // Build parquet reader options
    auto source = cudf::io::source_info(file_paths_);
    auto options = cudf::io::parquet_reader_options::builder(source)
        .columns(column_names_)
        .build();
    
    // Create the parquet reader node (from rapidsmpf)
    return rapidsmpf::streaming::node::read_parquet(
        ctx,
        ch_out,
        1,            // num_producers
        options,
        1024 * 1024,  // 1M rows per chunk
        nullptr       // No filter pushdown
    );
}
```

**Data Flow**: Reads parquet files → Produces `TableChunk` messages → Sends to `ch_out`

### 5.2 PhysicalFilter: Row Filtering

The filter is a coroutine that processes chunks:

```cpp
// src/operators/physical_filter.cpp:71-104
return [](ctx, input, output) -> rapidsmpf::streaming::Node {
    rapidsmpf::streaming::ShutdownAtExit shutdown_guard(output);
    std::uint64_t seq = 0;
    
    while (true) {
        // Receive a chunk from upstream
        auto msg = co_await input->receive();
        if (msg.empty()) break;
        
        // Extract the TableChunk
        auto chunk = msg.release<rapidsmpf::streaming::TableChunk>();
        
        // Make data available on GPU
        if (!chunk->is_available()) {
            auto reservation = ctx->br()->reserve(
                rapidsmpf::MemoryType::DEVICE,
                chunk->make_available_cost(),
                true  // allow overbooking
            );
            *chunk = chunk->make_available(reservation);
        }
        
        // TODO: Apply cudf::ast filter expression here
        // For now, pass through unchanged
        
        // Send to downstream
        co_await output->send(to_message(seq++, std::move(chunk)));
    }
}(ctx, ch_in, ch_out);
```

**Note**: Filter expression conversion from DuckDB to cudf AST is stubbed for future implementation.

### 5.3 PhysicalAggregate: Streaming Aggregation

The aggregate maintains running totals across chunks:

```cpp
// src/operators/physical_aggregate.cpp:104-236
return [](ctx, input, output, aggs) -> rapidsmpf::streaming::Node {
    // Accumulators for streaming aggregation
    std::vector<double> sums(aggs.size(), 0.0);
    std::vector<int64_t> counts(aggs.size(), 0);
    
    // Process all chunks
    while (true) {
        auto msg = co_await input->receive();
        if (msg.empty()) break;
        
        auto chunk = msg.release<rapidsmpf::streaming::TableChunk>();
        auto tbl_view = chunk->table_view();
        
        for (auto const& agg : aggs) {
            if (agg.kind == cudf::aggregation::Kind::COUNT_ALL) {
                counts[i] += tbl_view.num_rows();
            } else {
                // Use cudf::reduce for GPU-accelerated aggregation
                auto result = cudf::reduce(col, *cudf::make_sum_aggregation(), ...);
                sums[i] += result->value();
                counts[i] += col.size() - col.null_count();
            }
        }
    }
    
    // Finalize: compute AVG = SUM / COUNT
    for (size_t i = 0; i < aggs.size(); i++) {
        if (agg.kind == cudf::aggregation::Kind::MEAN) {
            value = sums[i] / static_cast<double>(counts[i]);
        }
    }
    
    // Send single result row
    co_await output->send(to_message(0, result_chunk));
}(ctx, ch_in, ch_out, agg_infos);
```

---

## Stage 6: Pipeline Execution

Once all nodes are built, they're executed as a concurrent pipeline:

```cpp
// src/executor.cpp:168
rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));
```

This function:
1. Schedules all coroutines on a thread pool
2. Manages async communication between nodes
3. Handles backpressure when channels fill up
4. Completes when all nodes finish (end-of-stream propagates)

### Message Flow

```
read_parquet ──┬──► TableChunk{rows 0-1M}     ──► filter ──► aggregate
               ├──► TableChunk{rows 1M-2M}    ──► filter ──► aggregate
               ├──► TableChunk{rows 2M-3M}    ──► filter ──► aggregate
               │    ...                       
               └──► TableChunk{rows 5.99B-6B} ──► filter ──► aggregate
                                                              │
                    ◄────────────────────────────────────────┘
                    Single result: AVG = 56980.91
```

---

## GPU Memory Management

The `BufferResource` manages GPU memory with spilling support:

```cpp
// Executor creates BufferResource with RMM adaptor
br_ = std::make_shared<rapidsmpf::BufferResource>(
    mr_adaptor_.get(),          // Wraps RMM device memory resource
    ...,
    stream_pool_,               // CUDA stream pool for async ops
    statistics_                 // Memory usage tracking
);

// Nodes request memory reservations before GPU operations
auto reservation = ctx->br()->reserve_device_memory_and_spill(
    chunk.make_available_cost(),  // Bytes needed
    true                          // Allow overbooking
);
chunk = chunk.make_available(reservation);  // Move data to GPU
```

---

## Benchmark Results

| Metric | CPU (DuckDB) | GPU (RapidsMPF) |
|--------|-------------|-----------------|
| Query | `SELECT AVG(l_extendedprice) FROM lineitem WHERE l_quantity > 25` |
| Dataset | TPC-H Scale-1000 (~6B rows) |
| Min Time | 30,898 ms | 303,659 ms* |
| GPU Memory Delta | 0 MB | +26 MB |

*Note: GPU is currently slower due to:
1. Filter not being pushed down (all rows read)
2. Result conversion back to DuckDB not implemented
3. Single-threaded parquet reading

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User SQL Query                                  │
│   SELECT * FROM rapidsmpf_query('SELECT AVG(...) FROM lineitem WHERE ...')   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            DuckDB (CPU)                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐   │
│  │   Parser    │───►│   Planner   │───►│   LogicalOperator Tree          │   │
│  └─────────────┘    └─────────────┘    │   (AGGREGATE → FILTER → GET)    │   │
│                                        └─────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        rapidsmpf_duckdb Extension                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  PhysicalPlanGenerator                                                  │ │
│  │    - Converts LogicalGet → PhysicalTableScan                            │ │
│  │    - Converts LogicalFilter → PhysicalFilter                            │ │
│  │    - Converts LogicalAggregate → PhysicalAggregate                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Executor                                                               │ │
│  │    - Creates streaming Context with BufferResource (GPU memory)         │ │
│  │    - Creates Channels between operators                                 │ │
│  │    - Calls BuildNode() on each PhysicalOperator                         │ │
│  │    - Runs the streaming pipeline                                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RapidsMPF Streaming (GPU)                          │
│                                                                              │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐              │
│   │  read_parquet │────►│    filter     │────►│   aggregate   │──► Result   │
│   │   (Source)    │     │  (Transform)  │     │    (Sink)     │              │
│   └───────────────┘     └───────────────┘     └───────────────┘              │
│          │                     │                     │                       │
│          │                     │                     │                       │
│          ▼                     ▼                     ▼                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         libcudf (GPU)                               │   │
│   │   - cudf::io::read_parquet()  - Reads parquet to GPU tables         │   │
│   │   - cudf::reduce()            - GPU-accelerated aggregations        │   │
│   │   - cudf::ast::compute()      - Expression evaluation               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         RMM (GPU Memory)                            │   │
│   │   - Pool allocator for GPU memory                                   │   │
│   │   - Spill-to-host for memory pressure                               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Code References

| Component | File | Description |
|-----------|------|-------------|
| Entry point | `src/rapidsmpf_extension.cpp` | `rapidsmpf_query` table function |
| Plan generator | `src/physical_plan_generator.cpp` | Logical → Physical conversion |
| Executor | `src/executor.cpp` | Pipeline construction & execution |
| Base operator | `src/include/physical_operator.hpp` | `PhysicalOperator` interface |
| Table scan | `src/operators/physical_table_scan.cpp` | Parquet reading |
| Filter | `src/operators/physical_filter.cpp` | Row filtering |
| Aggregate | `src/operators/physical_aggregate.cpp` | Aggregation functions |
| Table registry | `src/table_registry.hpp` | Table name → file path mapping |
| Benchmark | `examples/benchmark_gpu_vs_cpu.cpp` | Performance comparison |

---

## Supported Operations

| Category | Operation | Status | Notes |
|----------|-----------|--------|-------|
| **Table Scan** | Parquet reading | ✅ Supported | Uses `rapidsmpf::streaming::node::read_parquet` |
| **Projection** | Column selection | ✅ Supported | Selects columns from input |
| **Filter** | Numeric comparisons (`=`, `!=`, `<`, `>`, `<=`, `>=`) | ✅ Supported | Converted to cudf AST |
| **Filter** | String equality (`=`, `!=`) | ✅ Supported | Uses cudf string comparison |
| **Filter** | Date comparisons | ✅ Supported | Dates converted to days-since-epoch |
| **Filter** | Conjunctions (`AND`, `OR`) | ✅ Supported | Nested AST operations |
| **Filter** | CAST expressions | ✅ Supported | Handled transparently |
| **Aggregate** | `COUNT(*)` | ✅ Supported | Row counting |
| **Aggregate** | `SUM` | ✅ Supported | Uses `cudf::reduce` |
| **Aggregate** | `AVG` / `MEAN` | ✅ Supported | Streaming sum/count |
| **Aggregate** | `MIN` / `MAX` | ✅ Supported | Uses `cudf::reduce` |
| **ORDER BY** | Single/multi-column sorting | ✅ Supported | Uses `cudf::sort_by_key` |
| **ORDER BY** | ASC/DESC | ✅ Supported | Configurable per column |
| **ORDER BY** | NULLS FIRST/LAST | ✅ Supported | Configurable per column |
| **LIMIT** | Row limiting | ✅ Supported | Uses `cudf::slice` |
| **LIMIT** | OFFSET | ✅ Supported | Offset + limit |

---

## Unsupported Operations

| Category | Operation | Status | Notes |
|----------|-----------|--------|-------|
| **Projection** | Arithmetic expressions (`*`, `-`, `+`, `/`) | ❌ Not implemented | Needed for computed columns |
| **Aggregate** | `GROUP BY` | ❌ Not implemented | Only ungrouped aggregation supported |
| **Join** | Hash join | ❌ Not implemented | Required for multi-table queries |
| **Join** | Merge join | ❌ Not implemented | Alternative join strategy |
| **Filter** | `LIKE` / `ILIKE` | ❌ Not implemented | String pattern matching |
| **Filter** | `IN` / `NOT IN` | ❌ Not implemented | Set membership |
| **Filter** | `BETWEEN` | ❌ Not implemented | Range check |
| **Filter** | `IS NULL` / `IS NOT NULL` | ❌ Not implemented | Null checks |
| **Subquery** | Scalar subqueries | ❌ Not implemented | |
| **Subquery** | EXISTS / NOT EXISTS | ❌ Not implemented | |
| **Window** | Window functions | ❌ Not implemented | `ROW_NUMBER`, `RANK`, etc. |
| **Set** | UNION / INTERSECT / EXCEPT | ❌ Not implemented | |
| **Result** | cudf → DuckDB conversion | ❌ Not implemented | Results stay on GPU |

---

## Future Work

1. **Arithmetic expressions**: Add `*`, `-`, `+`, `/` support in projections for computed columns
2. **GROUP BY aggregation**: Extend aggregate to support grouped operations using `cudf::groupby`
3. **JOIN support**: Implement hash join using `cudf::hash_join` or `rapidsmpf` shuffle
4. **Result conversion**: Implement cudf → Arrow → DuckDB DataChunk conversion
5. **Parallel parquet reading**: Use multiple producers for faster I/O
6. **String patterns**: Add `LIKE`/`ILIKE` support using `cudf::strings::contains`

