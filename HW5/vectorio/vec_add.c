#include "vec_add.h"

#include "timer.h"
#include "util.h"
#include "opencl_util.h"

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel_normio;
static cl_kernel kernel_vecio;
static cl_mem gpu_mem_A;
static cl_mem gpu_mem_B;
static cl_mem gpu_mem_C;

void vec_add_init(int N) {
  cl_int err;

  // Get OpenCL platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_OPENCL(err);
  print_platform_info(platform);

  // Get OpenCL device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_OPENCL(err);
  print_device_info(device);

  // Create OpenCL context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_OPENCL(err);

  // Create OpenCL command queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_OPENCL(err);

  /*
   * Compile OpenCL program from "kernel.cl.c"
   * The name of kernel file is usually "kernel.cl",
   * but appending ".c" to the end of the filename helps text editors' syntax-highlighting.
   */
  program = create_and_build_program_with_source(context, device, "kernel.cl.c");

  kernel_normio = clCreateKernel(program, "vec_add_normal_io", &err); 
  CHECK_OPENCL(err);
  kernel_vecio = clCreateKernel(program, "vec_add_vector_io", &err); 
  CHECK_OPENCL(err);

  // Create GPU buffers for vectors
  gpu_mem_A = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
  CHECK_OPENCL(err);
  gpu_mem_B = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
  CHECK_OPENCL(err);
  gpu_mem_C = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
  CHECK_OPENCL(err);

  err = clFinish(queue);
  CHECK_OPENCL(err);
}

void vec_add_finalize() {
  // Free all resources we allocated
  clReleaseMemObject(gpu_mem_A);
  clReleaseMemObject(gpu_mem_B);
  clReleaseMemObject(gpu_mem_C);
  clReleaseKernel(kernel_normio);
  clReleaseKernel(kernel_vecio);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

void vec_add(float *A, float *B, float *C, int N) {
  cl_int err;

  // Vector A and B is currently on CPU. Send them to GPU.
  err = clEnqueueWriteBuffer(queue, gpu_mem_A, CL_TRUE, 0, N * sizeof(float), A, 0, NULL, NULL);
  CHECK_OPENCL(err);
  err = clEnqueueWriteBuffer(queue, gpu_mem_B, CL_TRUE, 0, N * sizeof(float), B, 0, NULL, NULL);
  CHECK_OPENCL(err);

  // Setup kernel arguments
  err = clSetKernelArg(kernel_normio, 0, sizeof(cl_mem), &gpu_mem_A);
  CHECK_OPENCL(err);
  err = clSetKernelArg(kernel_normio, 1, sizeof(cl_mem), &gpu_mem_B);
  CHECK_OPENCL(err);
  err = clSetKernelArg(kernel_normio, 2, sizeof(cl_mem), &gpu_mem_C);
  CHECK_OPENCL(err);
  err = clSetKernelArg(kernel_normio, 3, sizeof(int), &N);
  CHECK_OPENCL(err);

  err = clSetKernelArg(kernel_vecio, 0, sizeof(cl_mem), &gpu_mem_A);
  CHECK_OPENCL(err);
  err = clSetKernelArg(kernel_vecio, 1, sizeof(cl_mem), &gpu_mem_B);
  CHECK_OPENCL(err);
  err = clSetKernelArg(kernel_vecio, 2, sizeof(cl_mem), &gpu_mem_C);
  CHECK_OPENCL(err);
  err = clSetKernelArg(kernel_vecio, 3, sizeof(int), &N);
  CHECK_OPENCL(err);

  // Setup OpenCL global work size and local work size
  size_t gws[1] = {N/16}, lws[1] = {128};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // warm up
  err = clEnqueueNDRangeKernel(queue, kernel_normio, 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_OPENCL(err);
  err = clFinish(queue);
  CHECK_OPENCL(err);

  // Run kernels
  timer_start(0);
  err = clEnqueueNDRangeKernel(queue, kernel_normio, 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_OPENCL(err);
  err = clFinish(queue);
  CHECK_OPENCL(err);
  timer_stop(0);

  timer_start(1);
  err = clEnqueueNDRangeKernel(queue, kernel_vecio, 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_OPENCL(err);
  err = clFinish(queue);
  CHECK_OPENCL(err);
  timer_stop(1);

  // After running kernel, result resides in gpu_mem_C. Send it back to CPU.
  err = clEnqueueReadBuffer(queue, gpu_mem_C, CL_TRUE, 0, N * sizeof(float), C, 0, NULL, NULL);
  CHECK_OPENCL(err);

  err = clFinish(queue);
  CHECK_OPENCL(err);
}
