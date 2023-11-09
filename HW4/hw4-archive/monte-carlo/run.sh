#!/bin/bash

: ${NODES:=2}

salloc -N $NODES --partition class1 --exclusive      \
         mpirun --bind-to none -mca btl ^openib -npernode 1 \
         ./main $@
