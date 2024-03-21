#!/usr/bin/env bash
set -e
set -x

LIBDPU_BRANCH="shipped"
BUILDDIR=benchmark_$LIBDPU_BRANCH
rm -rf $BUILDDIR

echo "BRANCH: ${LIBDPU_BRANCH}"
mkdir -p $BUILDDIR

# Build DPU code
cd dpu
make -EBUILDDIR="../$BUILDDIR/" -j
cd ..

# Build Benchmark
cd $BUILDDIR
cmake -DCMAKE_BUILD_TYPE=Release -GNinja -DSHIPPED_LIBDPU=ON ..
ninja

export UPMEM_PROFILE_BASE=backend="hw"
mkdir -p data
host/checksum
host/benchmark >> ../data/$LIBDPU_BRANCH.log 2>> ../data/$LIBDPU_BRANCH.csv
