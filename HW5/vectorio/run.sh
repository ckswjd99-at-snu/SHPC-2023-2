#!/bin/bash

salloc  --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --partition=class1     \
         mpirun ./main 209715200
