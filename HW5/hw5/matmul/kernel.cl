__kernel void sgemm(__global float4 *A, __global float4 *B, __global float4 *C, int M, int N, int K) {
  // HYPERPARAMS
  const int TSM = 128;   // Local work size (M)
  const int TSN = 128;   // Local work size (N)
  const int TSK = 32;   // Local work size (K)
  const int WPTM = 1;    // Work per thread (M)
  const int WPTN = 8;    // Work per thread (N)
  const int VEC = 4;    // Vector size (4 floats)
  
  // IDENTIFY THREAD
  const int local_row = get_local_id(0);    // Local row ID (max: TSM/WPTM)
  const int local_col = get_local_id(1);    // Local col ID (max: TSN/WPTN/VEC)
  const int group_row = get_group_id(0);    // Work-group ID (0..M/TSM)
  const int group_col = get_group_id(1);    // Work-group ID (0..N/TSN)
  const int global_row = TSM * group_row + local_row * WPTM;        // Row LOC of C (0..M)
  const int global_col = TSN/VEC * group_col + local_col * WPTN;  // Col LOC of C (0..N/VEC)

  // LOCAL MEMS
  __local float4 local_A[TSM][TSK/VEC+2];   // Local memory to fit a tile of TSM*TSK elements of A
  __local float4 local_B[TSK][TSN/VEC];     // Local memory to fit a tile of TSK*TSN elements of B

  // ACCUMULATORS
  float4 acc[WPTM][WPTN];
  for (int wm=0; wm<WPTM; wm++)
    for (int wn=0; wn<WPTN; wn++)
      acc[wm][wn] = (0.0f, 0.0f, 0.0f, 0.0f);

  // LOOP OVER ALL TILES
  for (int tile=0; tile<K/TSK; tile++) {
    
    // Compressed load: A
    for (int wm=0; wm<WPTM; wm++)
      for (int wn=0; wn<WPTN * TSK / TSN; wn++) 
        local_A[WPTM * local_row + wm][(WPTN * TSK / TSN) * local_col + wn]
          = A[K/VEC * (global_row + wm) + TSK/VEC * tile + (WPTN * TSK / TSN) * local_col + wn];

    // Compressed load: B
    for (int wm=0; wm<WPTM; wm++)
      for (int wn=0; wn<WPTN; wn++) 
        local_B[(WPTM * local_row + wm) * TSK / TSM][local_col * WPTN + wn]
          = B[N/VEC * (TSK * tile + (WPTM * local_row + wm) * TSK / TSM) + global_col + wn];

    // Barrier for local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Multiplicate and accumulate
    for (int k=0; k<TSK; k++) {
      float val_A;
      for (int wm=0; wm<WPTM; wm++) {
        
        switch(k % VEC) {
          case 0: val_A = local_A[(WPTM * local_row + wm)][k/VEC].x; break;
          case 1: val_A = local_A[(WPTM * local_row + wm)][k/VEC].y; break;
          case 2: val_A = local_A[(WPTM * local_row + wm)][k/VEC].z; break;
          case 3: val_A = local_A[(WPTM * local_row + wm)][k/VEC].w; break;
        }
      
        for (int wn=0; wn<WPTN; wn++) 
          acc[wm][wn] += val_A * local_B[k][local_col * WPTN + wn];
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
