__kernel void sgemm(__global float4 *A, __global float4 *B, __global float4 *C, int M, int N, int K) {
  // HYPERPARAMS
  const int TSM = 128;   // Local work size (M)
  const int TSN = 128;   // Local work size (N)
  const int TSK = 32;   // Local work size (K)
  const int WPT = 8;    // Work per thread (1 x 4VEC = 1 x 16floats)
  const int VEC = 4;    // Vector size (4 floats)
  
  // IDENTIFY THREAD
  const int local_row = get_local_id(0);    // Local row ID (max: TSM)
  const int local_col = get_local_id(1);    // Local col ID (max: TSN/WPT/VEC)
  const int group_row = get_group_id(0);    // Work-group ID (0..M/TSM)
  const int group_col = get_group_id(1);    // Work-group ID (0..N/TSN)
  const int global_row = TSM * group_row + local_row;        // Row LOC of C (0..M)
  const int global_col = TSN/VEC * group_col + local_col * WPT;  // Col LOC of C (0..N/VEC)

  // LOCAL MEMS
  __local float4 local_A[TSM][TSK/VEC];   // Local memory to fit a tile of TSM*TSK elements of A
  __local float4 local_B[TSK][TSN/VEC];   // Local memory to fit a tile of TSK*TSN elements of B

  // ACCUMULATORS
  float4 acc[WPT];
  for (int w=0; w<WPT; w++) {
      acc[w] = (0.0f, 0.0f, 0.0f, 0.0f);
  }

  // LOOP OVER ALL TILES
  for (int tile=0; tile<K/TSK; tile++) {
    
    // Compressed load: A
    for (int w=0; w<WPT * TSK / TSN; w++) 
      local_A[local_row][(WPT * TSK / TSN) * local_col + w]
        = A[K/VEC * global_row + TSK/VEC * tile + (WPT * TSK / TSN) * local_col + w];

    // Compressed load: B
    for (int w=0; w<WPT; w++) 
      local_B[local_row * TSK / TSM][local_col * WPT + w]
        = B[N/VEC * (TSK * tile + local_row * TSK / TSM) + global_col + w];

    // Barrier for local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Multiplicate and accumulate
    for (int k=0; k<TSK; k++) {
      float val_A;
      switch(k % VEC) {
        case 0: val_A = local_A[local_row][k/VEC].x; break;
        case 1: val_A = local_A[local_row][k/VEC].y; break;
        case 2: val_A = local_A[local_row][k/VEC].z; break;
        case 3: val_A = local_A[local_row][k/VEC].w; break;
      }
       
      for (int w=0; w<WPT; w++) {
        acc[w] += val_A * local_B[k][local_col * WPT + w];
      }
    }

    // Barrier for local memory
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // WRITE RESULT
  for (int w=0; w<WPT; w++) {
    C[N/VEC * global_row + global_col + w] = acc[w];
  }
}
