#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# --- Configuration Variables ---
NUM_RANKS=8
BYTES_PER_ROW=4
NUM_COLUMNS=4
TOTAL_SIZE_GB=16  # Total input table size per rank in GB
APP_PATH=/datasets/bzaitlen/GitRepos/rapidsmpf/cpp/build/benchmarks/bench_shuffle
NSYS_BIN=/datasets/pentschev/nsight-systems-2025.5.1/bin/nsys

# --- Environment Variables (Set for the entire script execution) ---
export UCX_TLS=^sm
export FORCE_UCX_NET_DEVICES=unset
export UCX_MAX_RNDV_RAILS=1
export UCX_LOG_LEVEL=info
export UCX_RNDV_FRAG_SIZE=host:512K,cuda:32M
export UCX_RNDV_FRAG_ALLOC_COUNT=host:128,cuda:16
# export UCX_RNDV_FRAG_ALLOC_COUNT=host:128,cuda:128 # 128 fragments per staging buffer
# export UCX_RNDV_FRAG_SIZE=host:512K,cuda:4M # cuda:4MB default

# BINDER_ARGS is for the binder.sh script's internal configuration (8 ranks, 14 cores/rank)
export BINDER_ARGS="${NUM_RANKS} 14"


# --- Application Arguments ---
# Arguments for the benchmark executable that are common to both runs:
  # -C <comm>  Communicator {mpi, ucxx} (default: mpi)
  # -r <num>   Number of runs (default: 1)
  # -w <num>   Number of warmup runs (default: 0)
  # -c <num>   Number of columns in the input tables (default: 1)
  # -n <num>   Number of rows per rank (default: 1M)
  # -p <num>   Number of partitions (input tables) per rank (default: 1)
  # -o <num>   Number of output partitions per rank (default: 1)
  # -m <mr>    RMM memory resource {cuda, pool, async, managed} (default: pool)
  # -l <num>   Device memory limit in MiB (default:-1, disabled)
  # -i         Use `concat_insert` method, instead of `insert`.
  # -g         Use pre-partitioned (hash) input tables (default: unset, hash partition during insertion)
  # -s         Discard output chunks to simulate streaming (default: disabled)
  # -x         Enable memory profiler (default: disabled)
  # -h         Display this help message

# --- Conditional Profiling Logic ---
# Check if the first argument is '--with-nsys'
if [ "$1" == "--with-nsys" ]; then
    echo "Profiling enabled via --with-nsys. Using Nsight Systems."

    # Define the NSYS prefix with the unique rank output specifier
    NSYS_PREFIX="${NSYS_BIN} profile \
      -o shuffle.rapidsmpf-no-spill-16MB\
      -f true --cuda-memory-usage=true --cuda-event-trace=false \
      --nvtx-domain-exclude=CCCL,libkvikio \
      --stats=true -t cuda,ucx,nvtx"

    # Remove the '--with-nsys' argument so it's not passed to the application
    shift
else
    echo "Profiling disabled. Running standard execution."
    NSYS_PREFIX=""
fi


# --- Sweep over partition sizes ---
# Format: "PARTITION_SIZE_MB NUM_ROWS NUM_PARTITIONS"
# Calculations maintain 16GB total: NUM_ROWS * 4 bytes * 4 columns * NUM_PARTITIONS = 16GB
PARTITION_CONFIGS=(
    # "2 131072 8192"        # 2MB partitions
    # "16 1048576 1024"      # 16MB partitions
    # "64 4194304 256"       # 64MB partitions
    # "256 16777216 64"      # 256MB partitions
    # "512 33554432 32"      # 512MB partitions
    "768 50331648 21"      # 768MB partitions (21 partitions = 15.75GB, close to 16GB)
    # "1024 67108864 16"     # 1GB partitions
    # "2048 134217728 8"     # 2GB partitions
)

for CONFIG in "${PARTITION_CONFIGS[@]}"; do
    # Parse the configuration
    read PARTITION_SIZE_MB NUM_ROWS NUM_PARTITIONS <<< "$CONFIG"

    # Calculate actual total size for verification
    ACTUAL_SIZE_GB=$(echo "scale=2; $NUM_ROWS * 4 * 4 * $NUM_PARTITIONS / 1024 / 1024 / 1024" | bc)

    echo "======================================================================"
    echo "Running with partition size: ${PARTITION_SIZE_MB}MB"
    echo "  NUM_ROWS per partition: ${NUM_ROWS}"
    echo "  NUM_PARTITIONS: ${NUM_PARTITIONS}"
    echo "  Total size per rank: ${ACTUAL_SIZE_GB}GB"
    echo "======================================================================"

    # Base APP_ARGS with calculated values
    APP_ARGS="-C ucxx -w 10 -r 1 -g -s -x -b -p ${NUM_PARTITIONS} -o 4 -c ${NUM_COLUMNS} -n ${NUM_ROWS} -x -m async"

    # Output file for this partition size
    OUTPUT_FILE="no-spill-${PARTITION_SIZE_MB}MB-async-UCXFRAG_512K_32M_128_16.txt"

    # --- Execution Command ---
    echo "Executing Command:"
    echo "----------------------------------------------------------------------"
    echo "${NSYS_PREFIX} mpiexec -n ${NUM_RANKS} binder.sh ${APP_PATH} ${APP_ARGS} \"$@\""
    echo "Output will be saved to: ${OUTPUT_FILE}"
    echo "----------------------------------------------------------------------"

    # Actual execution with output redirection
    $NSYS_PREFIX mpiexec -n ${NUM_RANKS} \
      binder.sh ${APP_PATH} ${APP_ARGS} "$@" > "${OUTPUT_FILE}" 2>&1

    sleep 2
    echo "Completed run with partition size ${PARTITION_SIZE_MB}MB"
    echo ""
done

echo "======================================================================"
echo "All runs completed!"
echo "======================================================================"
