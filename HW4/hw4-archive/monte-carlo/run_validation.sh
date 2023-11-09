#!/bin/bash

NODES=2 ./run.sh -t 32 256
NODES=4 ./run.sh -t 32 512
NODES=2 ./run.sh -t 32 1024
NODES=4 ./run.sh -t 32 2048
NODES=2 ./run.sh -t 32 4096
NODES=4 ./run.sh -t 32 8192
NODES=2 ./run.sh -t 32 16384
NODES=4 ./run.sh -t 32 32768
NODES=2 ./run.sh -t 32 1000000
NODES=4 ./run.sh -t 32 10000000
