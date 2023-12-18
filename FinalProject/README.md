# Optimizing CNN Model with CUDA

## Applied Optimization Methods

- [x] Synchronously offload input to other nodes using MPI

- [x] Asynchronously offload input to other nodes using MPI

- [x] Calculate multiple batches at once

- [x] Calculate each operators with CUDA: `conv1d`, `layernorm`, `relu`, `maxpool1d`, `linear`, etc.
    - [x] Create CUDA version of each operators
        - `conv1d`: Rectangular blocking
        - `layernorm`: Naive
        - `relu`: All merged into the other operators
        - `maxpool1d`: Naive
        - `linear`: Naive
    - [x] Store most of intermediate features in global memory

- [x] Create weakly fused operators: `conv1d_relu`, `conv1d_stat`, `linear_relu`, etc.
    - [x] `conv1d_relu`: integrated into `conv1d`.
    - [x] `linear_relu`: integrated into `linear`.

- [ ] Create strongly fused operators (**no need to do this, it has come to be network-bottlenecked**)
    - [ ] `layernorm_relu_maxpool1d` (Conv block 1(back), Conv block 6(back))
    - [ ] `conv1d_relu_conv1d_relu_conv1d_relu_conv1d_stat` (Conv block 3 - Conv lbock 6(front))
    - [ ] `linear_relu_linear` (FC block 2 - FC block 3)

## Latency Breakdown

### Total Latency

Lower bound due to MPI & memory BW: 0.352305 sec

Measured in debug mode, in seconds.

- Elapsed time: **0.380468 sec**
- Throughput: **18533.932501 input(s)/sec**

| NODE              | 00(root) | 01       | 02       | 03       |
|:------------------|:--------:|:--------:|:--------:|:--------:|
| start_classifier  | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| init_mem          | 0.000000 | 0.000019 | 0.000033 | 0.000017 |
| init_ce           | 0.000206 | 0.000033 | 0.000068 | 0.000031 |
| scatter_start     | 0.000223 | 0.000217 | 0.000430 | 0.000224 |
| ce_start_gpu0     | 0.000260 | 0.009836 | 0.011851 | 0.012522 |
| ce_start_gpu1     | 0.000262 | 0.017100 | 0.019726 | 0.020753 |
| ce_start_gpu2     | 0.000305 | 0.024429 | 0.027037 | 0.028597 |
| ce_start_gpu3     | 0.000379 | 0.031801 | 0.034346 | 0.036484 |
| scatter_end       | 0.080509 | 0.251276 | 0.255178 | 0.279429 |
| ce_end            | 0.402353 | 0.382517 | 0.380537 | 0.379149 |
| gather_end        | 0.402579 | 0.382625 | 0.380742 | 0.379365 |
| finish_classifier | 0.402588 | 0.382635 | 0.380756 | 0.379378 |


### Computation Latency

- **Total latency**: 0.380468 sec
    - `conv1d_k3_cuda`: 0.028444 (7.48%)
    - `conv1d_k7_cuda`: 0.080676 (21.20%)
    - `linear_naive_cuda`: 0.000000 (0.00%)
    - `linear_reg_cuda`: 0.030356 (7.98%)
    - `layernorm_1008_cuda`: 0.041958 (11.03%)
    - `layernorm_102_cuda`: 0.014251 (3.75%)
    - `maxpool1d_k3_cuda`: 0.008116 (2.13%)
    - `argmax_f4_cuda`: 0.000000 (0.00%)
    - etc: 0.192874 (50.69%)


## Optimization History
- Baseline: 2.12 input(s)/sec
- Synchronous offload: 8.33 input(s)/sec
- Naively batched computation: 7.86 input(s)/sec
- Naive CUDA conv1d: 12.76 input(s)/sec
- Replace every conv1d with conv1d_cuda, fuse relu: 165.00 input(s)/sec
- Use multiple GPUs: 555.00 input(s)/sec
- Naive CUDA linear: 727.20 input(s)/sec
- Replace every linear with linear_cuda, fuse relu: 1152.75 input(s)/sec
- Merged maxpool1d and relu: 1290.74 input(s)/sec
- conv1d_k3 square blocking: 1505.14 input(s)/sec
- conv1d_k3 rectangular blocking: 1550.79 input(s)/sec
- conv1d hyperparameter tuning: 2537.34 input(s)/sec
- conv1d_k7 rectangular blocking: 3013.50 input(s)/sec
- Batched processing: 3501.90 input(s)/sec
- linear rectangular: 3644.37 input(s)/sec
- conv1d_k3, conv1d_k7 avoid bank conflict: 3753.42 input(s)/sec
- Naive linear normalization: 4241.36 input(s)/sec
- Naive maxpool1d: 5266.67 input(s)/sec
- Memory cleanup: 5865.32 input(s)/sec
- No more Tensor type: 6175.81 input(s)/sec
- Scatter into Scatterv: 5924.65 input(s)/sec
- Networking & offloading interleaved: 8587.53 input(s)/sec
- Fine-grained interleaving: 9303.47 input(s)/sec
- Asynchronous MPI: 13077.40 input(s)/sec
- Fine-grained layernorm_cuda: 15872.02 input(s)/sec
- Split gpu stream into mem and compute: 16359.99 input(s)/sec
- Avoid bank conflict + miscellaneous skills: 17077.22 input(s)/sec
- layernorm_cuda vectorized mem access: **19267.45** input(s)/sec

## Model Structure

```
{
    "conv_block_1": [
        conv1d(
            input=Tensor(BATCH_SIZE=N, VOCAB_SIZE=70, MAX_LENGTH=1014),
            weight=Tensor(OUT_CHANNEL=256, IN_CHANNEL=70, KERNEL_SIZE=7),
            bias=Tensor(OUT_CHANNEL=256),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=1008)
        ),
        layernorm(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=1008),
            gamma=(CHANNEL=256, MAX_LENGTH=1008),
            beta=(CHANNEL=256, MAX_LENGTH=1008),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=1008)
        ),
        relu(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=1008),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=1008)
        ),
        maxpool1d(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=1008),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=336),
            kernel_size=3,
            stride=3
        )
    ],
    "conv_block_2": [
        conv1d(
            input=Tensor(BATCH_SIZE=N, VOCAB_SIZE=256, MAX_LENGTH=336),
            weight=Tensor(OUT_CHANNEL=256, IN_CHANNEL=256, KERNEL_SIZE=7),
            bias=Tensor(OUT_CHANNEL=256),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=330)
        ),
        relu(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=330),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=330)
        ),
        maxpool1d(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=330),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=110)
        )
    ],
    "conv_block_3": [
        conv1d(
            input=Tensor(BATCH_SIZE=N, VOCAB_SIZE=256, MAX_LENGTH=110),
            weight=Tensor(OUT_CHANNEL=256, IN_CHANNEL=256, KERNEL_SIZE=3),
            bias=Tensor(OUT_CHANNEL=256),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=108)
        ),
        relu(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=108),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=108)
        )
    ],
    "conv_block_4": [
        conv1d(
            input=Tensor(BATCH_SIZE=N, VOCAB_SIZE=256, MAX_LENGTH=108),
            weight=Tensor(OUT_CHANNEL=256, IN_CHANNEL=256, KERNEL_SIZE=3),
            bias=Tensor(OUT_CHANNEL=256),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=106)
        ),
        relu(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=106),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=106)
        )
    ],
    "conv_block_5": [
        conv1d(
            input=Tensor(BATCH_SIZE=N, VOCAB_SIZE=256, MAX_LENGTH=106),
            weight=Tensor(OUT_CHANNEL=256, IN_CHANNEL=256, KERNEL_SIZE=3),
            bias=Tensor(OUT_CHANNEL=256),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=104)
        ),
        relu(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=104),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=104)
        )
    ],
    "conv_block_6": [
        conv1d(
            input=Tensor(BATCH_SIZE=N, VOCAB_SIZE=256, MAX_LENGTH=104),
            weight=Tensor(OUT_CHANNEL=256, IN_CHANNEL=256, KERNEL_SIZE=3),
            bias=Tensor(OUT_CHANNEL=256),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=102)
        ),
        layernorm(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=102),
            gamma=(CHANNEL=256, MAX_LENGTH=102),
            beta=(CHANNEL=256, MAX_LENGTH=102),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=102)
        ),
        relu(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=102),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=102)
        ),
        maxpool1d(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=102),
            output=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=34),
            kernel_size=3,
            stride=3
        )
    ],
    "collapse": [
        collapse(
            input=Tensor(BATCH_SIZE=N, CHANNEL=256, MAX_LENGTH=34),
            output=Tensor(BATCH_SIZE=N, CHANNEL=8704)
        )
    ],
    "fc_block_1": [
        linear(
            input=Tensor(BATCH_SIZE=N, CHANNEL=8704),
            weight=Tensor(OUT_CHANNEL=1024, IN_CHANNEL=8704),
            bias=Tensor(OUT_CHANNEL=1024),
            output=Tensor(BATCH_SIZE=N, CHANNEL=1024)
        ),
        relu(
            input=Tensor(BATCH_SIZE=N, CHANNEL=1024),
            output=Tensor(BATCH_SIZE=N, CHANNEL=1024),
        )
    ],
    "fc_block_2": [
        linear(
            input=Tensor(BATCH_SIZE=N, CHANNEL=1024),
            weight=Tensor(OUT_CHANNEL=1024, IN_CHANNEL=1024),
            bias=Tensor(OUT_CHANNEL=1024),
            output=Tensor(BATCH_SIZE=N, CHANNEL=1024)
        ),
        relu(
            input=Tensor(BATCH_SIZE=N, CHANNEL=1024),
            output=Tensor(BATCH_SIZE=N, CHANNEL=1024),
        )
    ],
    "fc_block_3": [
        linear(
            input=Tensor(BATCH_SIZE=N, CHANNEL=1024),
            weight=Tensor(OUT_CHANNEL=4, IN_CHANNEL=1024),
            bias=Tensor(OUT_CHANNEL=4),
            output=Tensor(BATCH_SIZE=N, CHANNEL=4)
        )
    ],
}
```