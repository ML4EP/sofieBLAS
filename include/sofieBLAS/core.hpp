#pragma once

#include <cstddef>

namespace traits {
template <typename TTag> class sofieBLAS;
}

template <typename TTag>
using sofieBLAS = typename traits::sofieBLAS<TTag>::Impl;

struct BlasOptions {
  std::size_t cacheLimit = 0;
  bool quantization = false;
  // GPU backends: run every quantized GEMM with the native kernels
  bool forceNative = false;
  // GPU backends: run every quantized GEMM with the vendor library's kernels,
  // skipping the startup probe and the known-mismatch workarounds.
  // Cannot be combined with forceNative.
  bool forceVendor = false;
};

enum class Epilogue { Default, Bias, ReluBias, GeluBias, Relu, Gelu };
