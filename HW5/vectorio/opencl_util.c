#include "opencl_util.h"

#include <stdio.h>

void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_OPENCL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_OPENCL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("OpenCL platform name: %s\n", buf);
  free(buf);
}

void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_OPENCL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_OPENCL(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("OpenCL device name: %s\n", buf);
  free(buf);
}

cl_program create_and_build_program_with_source(cl_context context, cl_device_id device, const char *file_name) {
  cl_int err;

  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char*)malloc(source_size + 1);
  size_t ret = fread(source_code, sizeof(char), source_size, file);
  if (ret != source_size) {
    printf("Failed to read %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  source_code[source_size] = '\0';
  fclose(file);

  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
  CHECK_OPENCL(err);
  free(source_code);

  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_OPENCL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
    char *log = (char*)malloc(log_size + 1);
    CHECK_OPENCL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_OPENCL(err);

  return program;
}
