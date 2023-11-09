#!/bin/bash

NODES=1 ./run.sh -n 10 -v -t 32 4096 4096 4096
NODES=2 ./run.sh -n 10 -v -t 32 8192 4096 4096
NODES=4 ./run.sh -n 10 -v -t 32 16384 4096 4096
