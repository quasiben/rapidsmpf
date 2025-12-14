# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=rapidsmpf_duckdb
EXT_CONFIG=${PROJ_DIR}extension_config.cmake
GEN=ninja

# Use the sirius duckdb source
DUCKDB_SRCDIR=${PROJ_DIR}../sirius/duckdb/

# Include the Makefile from extension-ci-tools
include ${PROJ_DIR}../sirius/extension-ci-tools/makefiles/duckdb_extension.Makefile

