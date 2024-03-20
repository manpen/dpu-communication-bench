#!/usr/bin/env bash

run_bench() {
    cd upmem-libdpu
    git checkout $1
    cd ..
    ./run_benchmark.sh $2
}

run_benchmark "state/original" ".*"
run_benchmark "state/nogather" ".*"
run_benchmark "state/scalar_prefetch" ".*"
run_benchmark "state/broadcast" ".*Broadcast.*"
run_benchmark "state/read" "Gather"
run_benchmark "state/write_matrix_transpose" ".*"
./run_shipped_benchmark
