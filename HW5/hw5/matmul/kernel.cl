__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  // HYPERPARAMS
  const int TS = 32;  // Local work size (32x32)
  
  // IDENTIFY THREAD
  const int local_row = get_local_id(0);    // Local row ID (max: TS)
  const int local_col = get_local_id(1);    // Local col ID (max: TS)
  const int global_row = get_global_id(0);  // Row ID of C (0..M)
  const int global_col = get_global_id(1);  // Col ID of C (0..N)

  // Compute a single element (loop over K)
  float acc = 0.0f;
  for (int k=0; k<K; k++) {
    acc += A[K * global_row + k] * B[N * k + global_col];
  }

  // Store the result
  C[global_row * N + global_col] = acc;
}
