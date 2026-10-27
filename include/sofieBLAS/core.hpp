#pragma once

#include <cstddef>

namespace traits {
template <typename TTag> class sofieBLAS;
}

template <typename TTag>
using sofieBLAS = typename traits::sofieBLAS<TTag>::Impl;

enum class Epilogue { Default, Bias, ReluBias, GeluBias, Relu };
enum class DataType { F32, I8, I32 };
struct BlasOptions {
  std::size_t cacheLimit = 0;
  bool quantization = false;
  bool forceNative = false;
};

enum class Epilogue { Default, Bias, ReluBias, GeluBias, Relu, Gelu };
