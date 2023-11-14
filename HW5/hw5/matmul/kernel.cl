__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  // HYPERPARAMS
  const int TS = 8;  // Local work size (32x32)
  
  // IDENTIFY THREAD
  const int local_row = get_local_id(0);    // Local row ID (max: TS)
  const int local_col = get_local_id(1);    // Local col ID (max: TS)
  const int group_row = get_group_id(0);    // Work-group ID (0..M/TS)
  const int group_col = get_group_id(1);    // Work-group ID (0..N/TS)
  const int global_row = TS * group_row + local_row;  // Row ID of C (0..M)
  const int global_col = TS * group_col + local_col;  // Col ID of C (0..N)

  // LOCAL MEMS
  __local float local_A[TS][TS];   // Local memory to fit a tile of TS*TS elements of A
  __local float local_B[TS][TS];   // Local memory to fit a tile of TS*TS elements of B

  // COMPUTE ACCUMULATOR
  float acc = 0.0f;   // Accumulator of one thread

  for (int tile=0; tile<K/TS; tile++) {
    // Load one tile of A and B into local memory
    local_A[local_row][local_col] = A[K * global_row + (tile * TS + local_col)];
    local_B[local_row][local_col] = B[N * (tile * TS + local_row) + global_col];

    // Synchronize to make sure the tile is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the computation for a single tile
    for (int k=0; k<TS; k++) {
      acc += local_A[local_row][k] * local_B[k][local_col];
    }

    // Synchronize before loading the next tile
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Store the result
  C[N * global_row + global_col] = acc;
}
