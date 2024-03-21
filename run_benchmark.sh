#!/usr/bin/env bash
set -e
set -x

LIBDPU_BRANCH=$(cd upmem-libdpu; git rev-parse --abbrev-ref HEAD | perl -pe 's/\//_/g')
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
cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja

ln -sf /usr/share/ upmem-libdpu/share
ln -sf $(pwd)/upmem-libdpu/hw/libdpuhw.so upmem-libdpu/api/

export UPMEM_PROFILE_BASE=backend="hw"
mkdir -p data
host/checksum
host/benchmark $1 > ../data/$LIBDPU_BRANCH.log 2> ../data/$LIBDPU_BRANCH.csv
