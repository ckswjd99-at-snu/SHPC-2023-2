__kernel void sgemm(__global float4 *A, __global float4 *B, __global float4 *C, int M, int N, int K) {
  // HYPERPARAMS
  const int VEC = 4;    // Vector size (4 floats)

  const int TSM = 128;   // Local work size (M)
  const int TSN = 128;   // Local work size (N)
  const int TSK = 16;   // Local work size (K)
  const int TSNV = TSN/VEC;
  const int TSKV = TSK/VEC;
  const int WPTM = 1;    // Work per thread (M)
  const int WPTN = 8;    // Work per thread (N)

  const int TOTAL_LOAD_A = TSM * TSK / VEC; // Total data load per workgroup (A)
  const int TOTAL_LOAD_B = TSK * TSN / VEC; // Total data load per workgroup (B)

  const int THREAD_LOAD_A = TOTAL_LOAD_A / (TSM * TSK / WPTM / WPTN); // Data load per thread (A)
  const int THREAD_LOAD_B = TOTAL_LOAD_B / (TSK * TSN / WPTM / WPTN); // Data load per thread (B)
  
  // IDENTIFY THREAD
  const int local_row = get_local_id(0);    // Local row ID (max: TSM/WPTM)
  const int local_col = get_local_id(1);    // Local col ID (max: TSN/WPTN/VEC)
  const int local_thread_idx = local_row * (TSN / WPTN / VEC) + local_col;
  const int group_row = get_group_id(0);    // Work-group ID (0..M/TSM)
  const int group_col = get_group_id(1);    // Work-group ID (0..N/TSN)
  const int global_row = TSM * group_row + local_row * WPTM;        // Row LOC of C (0..M)
  const int global_col = TSN/VEC * group_col + local_col * WPTN;  // Col LOC of C (0..N/VEC)

  // LOCAL MEMS
  __local float4 local_A[TSM][TSK/VEC+2];   // Local memory to fit a tile of TSM*TSK elements of A
  __local float4 local_B[TSK][TSN/VEC];     // Local memory to fit a tile of TSK*TSN elements of B

  // REGISTER
  float val_A[WPTM];
  float4 val_B;
  float4 acc[WPTM][WPTN];

  // INIT
  for (int wm=0; wm<WPTM; wm++)
    for (int wn=0; wn<WPTN; wn++)
      acc[wm][wn] = (0.0f, 0.0f, 0.0f, 0.0f);

  // LOOP OVER ALL TILES
  for (int tile=0; tile<K/TSK; tile++) {
    
    // Compressed load: A
    for (int wm=0; wm<WPTM; wm++) {
      volatile int local_row_idx = WPTM * local_row + wm;
      for (int wn=0; wn<WPTN * TSK / TSN; wn++) {
        int col_idx = (WPTN * TSK / TSN) * local_col + wn;
        local_A[local_row_idx][col_idx]
          = A[K/VEC * (global_row + wm) + TSK/VEC * tile + col_idx];
      }
    }

    // Compressed load: B
    for (int wm=0; wm<WPTM; wm++) {
      volatile int local_row_idx = (WPTM * local_row + wm) * TSK / TSM;
      volatile int global_row_offset = N/VEC * (TSK * tile + local_row_idx);
      for (int wn=0; wn<WPTN; wn++) 
        local_B[local_row_idx][local_col * WPTN + wn] = B[global_row_offset + global_col + wn];
    }

    // Barrier for local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Multiplicate and accumulate
    for (int k=0; k<TSK; k++) {
      for (int wm=0; wm<WPTM; wm++) {
        val_A[wm] = local_A[WPTM * local_row + wm][k/VEC][k % VEC];
      }

      for (int wn=0; wn<WPTN; wn++) {
        val_B = local_B[k][local_col * WPTN + wn];
        for (int wm=0; wm<WPTM; wm++) 
          acc[wm][wn] += val_A[wm] * val_B;

      }
      
    }

    // Barrier for local memory
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // WRITE RESULT
  for (int wm=0; wm<WPTM; wm++)
    for (int wn=0; wn<WPTN; wn++)
      C[N/VEC * (global_row + wm) + global_col + wn] = acc[wm][wn];
  
}
