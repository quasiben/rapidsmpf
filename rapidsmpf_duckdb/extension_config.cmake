# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# This file is included by DuckDB's build system. It specifies which extension to load

# Extension from this repo
duckdb_extension_load(rapidsmpf_duckdb
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
    EXTENSION_VERSION dev
)

