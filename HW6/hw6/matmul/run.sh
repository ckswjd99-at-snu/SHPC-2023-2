#!/bin/bash

: ${NODES:=1}

salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
  mpirun --bind-to none -mca btl ^openib -npernode 1         \
  numactl --physcpubind 0-31                                 \
  ./main $@
