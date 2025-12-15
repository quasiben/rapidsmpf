#!/usr/bin/env python3
"""Create test parquet file for GPU execution test."""

import cudf
import numpy as np

# Create test data
n_rows = 100
df = cudf.DataFrame({
    'id': np.arange(n_rows, dtype=np.int64),
    'value': np.arange(n_rows, dtype=np.float64) * 1.5
})

# Write to parquet
output_path = '/tmp/rapidsmpf_substrait_test.parquet'
df.to_parquet(output_path, index=False)
print(f"Created {output_path} with {len(df)} rows")
print(df.head())

