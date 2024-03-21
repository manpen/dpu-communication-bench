#!/usr/bin/env bash
set -e

BUILDDIR=build_benchmark
rm -rf $BUILDDIR
mkdir $BUILDDIR

# Make sure library repo is available
[ ! -d "upmem-libdpu" ] && git clone https://github.com/manpen/upmem-libdpu.git
mkdir -p data

# Build DPU code
cd dpu
make -EBUILDDIR="../$BUILDDIR/" -j
cd ..

# Setup Build enviroment
cd $BUILDDIR
cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja

ln -sf /usr/share/ upmem-libdpu/share
ln -sf $(pwd)/upmem-libdpu/hw/libdpuhw.so upmem-libdpu/api/
cd ..

# Run benchmarks
for repeat in 0 1 2 3 4;
do
  for b in unmodified replace_avxgather write_512b_to_bank write_to_rank_broadcast experimental_read;
  do
    cd upmem-libdpu
    git checkout feature/$b
    cd ..

    cd $BUILDDIR
    ninja
    host/benchmark >> ../data/$b.log 2>> ../data/$b.csv
    cd ..
  done

  ./run_shipped_benchmark.sh
done
