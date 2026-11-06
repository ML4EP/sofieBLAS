#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "sofieBLAS/backends/native/sofieBLAS_native.hpp"
#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>

template <typename TTag, typename TQueue> class FallbackBlas {
public:
  using Native = NativeBlas<TTag, TQueue>;

  explicit FallbackBlas(TQueue &queue) : native{queue} {}

  // the epilogue of a call as the native kernels take it
  template <typename TBias>
  static EpilogueParams<TBias>
  epilogueParams(Epilogue epilogue, float alpha, const float *alphaVector,
                 const float *scaleA, const float *scaleB, const float *scaleD,
                 const TBias *bias) {
    return Native::epilogueParams(epilogue, alpha, alphaVector, scaleA, scaleB,
                                  scaleD, bias);
  }

  template <typename TSafe>
  void int8Int32(TSafe &&safe, char transa, char transb, unsigned int m,
                 unsigned int n, unsigned int k, const std::int8_t *A,
                 long long strideA, const std::int8_t *B, long long strideB,
                 std::int32_t *C, long long strideC, int batchCount) {
    if (safe.int8Int32(transa, transb, m, n, k, A, strideA, B, strideB, C,
                       strideC, batchCount))
      return;
    native.template gemm<std::int32_t>(transa, transb, m, n, k, A, strideA, B,
                                       strideB, C, strideC, batchCount,
                                       EpilogueParams<float>{});
  }

  template <typename TSafe, typename TBias>
  void int8Int8(TSafe &&safe, char transa, char transb, unsigned int m,
                unsigned int n, unsigned int k, const std::int8_t *A,
                long long strideA, const std::int8_t *B, long long strideB,
                std::int8_t *C, long long strideC, int batchCount,
                EpilogueParams<TBias> const &params) {
    const long long mn = static_cast<long long>(m) * n;
    if (safe.int8Int32Available()) {
      auto *sum = static_cast<std::int32_t *>(native.scratch(
          static_cast<std::size_t>(mn) * batchCount * sizeof(std::int32_t)));
      if (safe.int8Int32(transa, transb, m, n, k, A, strideA, B, strideB, sum,
                         mn, batchCount)) {
        native.epilogue(sum, C, m, n, batchCount, strideC, params);
        return;
      }
    }
    native.template gemm<std::int32_t>(transa, transb, m, n, k, A, strideA, B,
                                       strideB, C, strideC, batchCount, params);
  }

  template <typename TSafe, typename TFp8, typename TOut, typename TBias>
  void fp8(TSafe &&safe, char transa, char transb, unsigned int m,
           unsigned int n, unsigned int k, const TFp8 *A, long long strideA,
           const TFp8 *B, long long strideB, TOut *C, long long strideC,
           int batchCount, EpilogueParams<TBias> const &params) {
    using Half = typename std::decay_t<TSafe>::Half;
    if (safe.fp16Fp32Available()) {
      const std::size_t mn = std::size_t{m} * n;
      const std::size_t countA =
          spanOf(strideA, std::size_t{m} * k, batchCount);
      const std::size_t countB =
          spanOf(strideB, std::size_t{k} * n, batchCount);
      const std::size_t bytesA = align(countA * sizeof(Half));
      const std::size_t bytesB = align(countB * sizeof(Half));
      auto *base = static_cast<std::byte *>(
          native.scratch(bytesA + bytesB + mn * batchCount * sizeof(float)));
      auto *hA = reinterpret_cast<Half *>(base);
      auto *hB = reinterpret_cast<Half *>(base + bytesA);
      auto *sum = reinterpret_cast<float *>(base + bytesA + bytesB);
      native.convert(A, hA, countA);
      native.convert(B, hB, countB);
      if (safe.fp16Fp32(transa, transb, m, n, k, static_cast<const Half *>(hA),
                        strideA, static_cast<const Half *>(hB), strideB, sum,
                        static_cast<long long>(mn), batchCount)) {
        native.epilogue(sum, C, m, n, batchCount, strideC, params);
        return;
      }
    }
    native.template gemm<float>(transa, transb, m, n, k, A, strideA, B, strideB,
                                C, strideC, batchCount, params);
  }

  Native native;

private:
  static std::size_t align(std::size_t bytes) {
    return (bytes + 255) / 256 * 256;
  }

  static std::size_t spanOf(long long stride, std::size_t count,
                            int batchCount) {
    return std::max(count, static_cast<std::size_t>(stride) *
                                   static_cast<std::size_t>(batchCount - 1) +
                               count);
  }
};
