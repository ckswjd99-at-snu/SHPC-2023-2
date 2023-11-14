__kernel void vec_add_normal_io(__global float *A,
                                __global float *B,
                                __global float *C,
                                int N) {
  int i = get_global_id(0) * 16;
  if (i >= N) return;

  for (int j = 0; j < 16; ++j) {
    C[i+j] = A[i+j] + B[i+j];
  }
}

__kernel void vec_add_vector_io(__global float *A,
                                __global float *B,
                                __global float *C,
                                int N) {
  int i = get_global_id(0);
  if (i >= N/16) return;

  float16 a = vload16(i, A);
  float16 b = vload16(i, B);
  float16 c = a + b;
  vstore16(c, i, C);
}
