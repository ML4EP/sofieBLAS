#pragma once

namespace traits {
template <typename TTag> class sofieBLAS;
}

template <typename TTag>
using sofieBLAS = typename traits::sofieBLAS<TTag>::Impl;

enum class Epilogue { Default, Bias, ReluBias, GeluBias, Relu };
enum class DataType { F32, I8, I32 };
