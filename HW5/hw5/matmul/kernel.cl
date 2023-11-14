__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  // HYPERPARAMS
  const int TS = 64;  // Local work size (32x32)
  const int WPT = 16;  // Work per thread (1x8)
  
  // IDENTIFY THREAD
  const int local_row = get_local_id(0);    // Local row ID (max: TS)
  const int local_col = get_local_id(1);    // Local col ID (max: TS)
  const int group_row = get_group_id(0);    // Work-group ID (0..M/TS)
  const int group_col = get_group_id(1);    // Work-group ID (0..N/TS)
  const int global_row = TS * group_row + local_row;        // Row ID of C (0..M)
  const int global_col = TS * group_col + local_col * WPT;  // Col ID of C (0..N)

  // LOCAL MEMS
  __local float local_A[TS][TS];   // Local memory to fit a tile of TS*TS elements of A
  __local float local_B[TS][TS];   // Local memory to fit a tile of TS*TS elements of B

  // ACCUMULATORS
  float acc[WPT];
  for (int w=0; w<WPT; w++) {
      acc[w] = 0.0f;
  }

  // LOOP OVER ALL TILES
  for (int tile=0; tile<K/TS; tile++) {
    
    for (int w=0; w<WPT; w++) {
      local_A[local_row][local_col * WPT + w] = A[M * global_row + TS * tile + local_col * WPT + w];
      local_B[local_row][local_col * WPT + w] = B[N * (TS * tile + local_row) + global_col + w];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k=0; k<TS; k++) {
      for (int w=0; w<WPT; w++) {
        acc[w] += local_A[local_row][k] * local_B[k][local_col * WPT + w];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // WRITE RESULT
  for (int w=0; w<WPT; w++) {
    C[N * global_row + global_col + w] = acc[w];
  }
}
