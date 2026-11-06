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
  bool forceNative = false;
};

enum class Epilogue { Default, Bias, ReluBias, GeluBias, Relu, Gelu };
