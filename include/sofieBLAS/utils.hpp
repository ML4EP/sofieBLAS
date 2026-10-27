#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace utils {

template <class Api>
[[noreturn]] void reportQuantBatchedMissing(unsigned int m, unsigned int n,
                                            unsigned int k, int batchCount,
                                            long long strideA,
                                            long long strideB) {
  std::cerr << "[sofieBLAS] No suitable " << Api::name
            << " algorithm found for the quantized batched multiply m=" << m
            << " n=" << n << " k=" << k << " batchCount=" << batchCount
            << " strideA=" << strideA << " strideB=" << strideB << "\n";
  std::exit(EXIT_FAILURE);
}

template <class Api>
[[noreturn]] void
reportAlgoMissing(typename Api::Operation transA,
                  typename Api::Operation transB,
                  typename Api::Epilogue epilogue,
                  const std::pair<std::size_t, std::size_t> &shapeA,
                  const std::pair<std::size_t, std::size_t> &shapeB) {
  std::cerr << "[sofieBLAS] No suitable " << Api::name
            << " algorithm found for " << "transA=" << transA
            << " transB=" << transB << " epilogue=" << epilogue << " A=["
            << shapeA.first << "x" << shapeA.second << "]" << " B=["
            << shapeB.first << "x" << shapeB.second << "]\n";
  std::exit(EXIT_FAILURE);
}

} // namespace utils
