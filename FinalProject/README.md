# Optimizing CNN Model with CUDA

## Optimization Methods to Apply

- [x] Synchronously offload input to other nodes using MPI

- [ ] Asynchronously offload input to other nodes using MPI

- [x] Calculate multiple batches at once

- [ ] Calculate each operators with CUDA: `conv1d`, `layernorm`, `relu`, `maxpool1d`, `linear`, etc.
    - [ ] Create CUDA version of each operators
        - `conv1d`: Naive
        - `layernorm`: None
        - `relu`: None
        - `maxpool1d`: None
        - `linear`: Naive
    - [ ] Store most of intermediate features in global memory

- [ ] Create weakly fused operators: `conv1d_relu`, `conv1d_stat`, `linear_relu`, etc.
    - [x] `conv1d_relu`: integrated into `conv1d`.
    - [x] `linear_relu`: integrated into `linear`.

- [ ] Create strongly fused operators
    - [ ] `layernorm_relu_maxpool1d` (Conv block 1(back), Conv block 6(back))
    - [ ] `conv1d_relu_conv1d_relu_conv1d_relu_conv1d_stat` (Conv block 3 - Conv lbock 6(front))
    - [ ] `linear_relu_linear` (FC block 2 - FC block 3)

## Optimization History
- Baseline: 2.12 input(s)/sec
- Synchronous offload: 8.33 input(s)/sec
- Naively batched computation: 7.86 input(s)/sec
- Naive CUDA conv1d: 12.76 input(s)/sec
- Replace every conv1d with conv1d_kernel, fuse relu: 165.00 input(s)/sec
- Use multiple GPUs: 555.00 input(s)/sec
- Naive CUDA linear: 727.20 input(s)/sec

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