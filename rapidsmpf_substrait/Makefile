# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# Makefile for rapidsmpf_substrait

.PHONY: all release debug clean configure help

BUILD_DIR = build
RELEASE_DIR = $(BUILD_DIR)/release
DEBUG_DIR = $(BUILD_DIR)/debug

# Number of parallel jobs (default to number of CPUs)
JOBS ?= $(shell nproc)

# Default target
all: release

# Release build
release: $(RELEASE_DIR)/Makefile
	@echo "Building release..."
	cmake --build $(RELEASE_DIR) -j$(JOBS)

$(RELEASE_DIR)/Makefile:
	@mkdir -p $(RELEASE_DIR)
	cd $(RELEASE_DIR) && cmake ../.. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Debug build
debug: $(DEBUG_DIR)/Makefile
	@echo "Building debug..."
	cmake --build $(DEBUG_DIR) -j$(JOBS)

$(DEBUG_DIR)/Makefile:
	@mkdir -p $(DEBUG_DIR)
	cd $(DEBUG_DIR) && cmake ../.. \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Configure only (useful for IDE integration)
configure:
	@mkdir -p $(RELEASE_DIR)
	cd $(RELEASE_DIR) && cmake ../.. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	@echo "Compile commands available at: $(RELEASE_DIR)/compile_commands.json"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Help
help:
	@echo "rapidsmpf_substrait Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build release (default)"
	@echo "  release  - Build release configuration"
	@echo "  debug    - Build debug configuration"
	@echo "  configure- Configure CMake (for IDE integration)"
	@echo "  clean    - Remove build directory"
	@echo "  help     - Show this message"
	@echo ""
	@echo "Variables:"
	@echo "  JOBS     - Number of parallel build jobs (default: nproc)"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Build rapidsmpf first: cd ../cpp && cmake -B build && cmake --build build"
	@echo "  - Ensure protobuf is installed"
	@echo ""
	@echo "Examples:"
	@echo "  make release         # Build release"
	@echo "  make debug           # Build debug"
	@echo "  make JOBS=4 release  # Build with 4 parallel jobs"

