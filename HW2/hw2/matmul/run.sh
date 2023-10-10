#!/bin/bash

srun --nodes=1 --exclusive --partition=class1 ./main $@
