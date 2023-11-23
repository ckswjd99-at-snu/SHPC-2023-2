/* HYPERPARAMS */
#define VEC 4
#define TSM 128
#define TSN 128
#define TSK 16

#define TSNV TSN/VEC
#define TSKV TSK/VEC

#define WPTM 1
#define WPTN 8

#define TOTAL_LOAD_A TSM * TSK / VEC
#define TOTAL_LOAD_B TSK * TSN / VEC

#define THREAD_LOAD_A TOTAL_LOAD_A / (TSM * TSK / WPTM / WPTN)
#define THREAD_LOAD_B TOTAL_LOAD_B / (TSK * TSN / WPTM / WPTN)

__kernel void sgemm_regular(__global float4 *A, __global float4 *B, __global float4 *C, int M, int N, int K) {
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
      for (int wm=0; wm<WPTM; wm++)
        val_A[wm] = local_A[WPTM * local_row + wm][k/VEC][k % VEC];

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

__kernel void sgemm_naive(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  // IDENTIFY THREAD
  const int local_row = get_local_id(0);    // Local row ID (max: TSM/WPTM)
  const int local_col = get_local_id(1);    // Local col ID (max: TSN/WPTN/VEC)
  const int group_row = get_group_id(0);    // Work-group ID (0..M/TSM)
  const int group_col = get_group_id(1);    // Work-group ID (0..N/TSN)

  int tile_M, tile_N;

  if (group_row == M/TSM) tile_M = M % TSM;
  else tile_M = TSM;

  if (group_col == N/TSN) tile_N = N % TSN;
  else tile_N = TSN;

  // COMPUTE
  const int TS = 16;
  int wm_start = tile_M * local_row / TS;
  int wm_end = tile_M * (local_row + 1) / TS;
  int wn_start = tile_N * local_col / TS;
  int wn_end = tile_N * (local_col + 1) / TS;

  for (int wm=wm_start; wm<wm_end; wm++) {
    for (int wn=wn_start; wn<wn_end; wn++) {
      float acc = 0.0f;
      for (int k=0; k<K; k++) {
        acc += A[K * (group_row * TSM + wm) + k] * B[N * k + group_col * TSN + wn];
      }
      C[N * (group_row * TSM + wm) + group_col * TSN + wn] = acc;
    }
  }

}

__kernel void sgemm(__global float4 *A, __global float4 *B, __global float4 *C, int M, int N, int K) {
  if (M % TSM == 0 && N % TSN == 0 && K % TSK == 0) {
    sgemm_regular(A, B, C, M, N, K);
  }
  else {
    sgemm_naive(A, B, C, M, N, K);
  }
}