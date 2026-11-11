// Shared by the SO3LR-Flash CUDA libraries' FFI handlers: the buffer types, the device's SM
// count, a kernel's dynamic shared-memory limit set once per device, and a launch's status.
#pragma once
#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using F32 = ffi::Buffer<ffi::F32>;
using F64 = ffi::Buffer<ffi::F64>;
using S32 = ffi::Buffer<ffi::S32>;
using RF32 = ffi::ResultBuffer<ffi::F32>;
using RS32 = ffi::ResultBuffer<ffi::S32>;

static int num_sms() {
  static int n = [] {
    int dev = 0, v = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
    return v;
  }();
  return n;
}

// the kernel's dynamic shared memory limit: the attribute is per device, set once on each
template <auto kernel>
static void set_max_smem(int bytes) {
  static bool done[64] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 64 && !done[dev]) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    done[dev] = true;
  }
}

static ffi::Error launch_status() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
  return ffi::Error::Success();
}
