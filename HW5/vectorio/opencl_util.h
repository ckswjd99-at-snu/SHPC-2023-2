#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define CHECK_OPENCL(err) \
  do { \
    if (err != CL_SUCCESS) { \
      printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      exit(EXIT_FAILURE); \
    } \
  } while (0); \

void print_platform_info(cl_platform_id platform);

void print_device_info(cl_device_id device);

cl_program create_and_build_program_with_source(cl_context context, cl_device_id device, const char *file_name);



