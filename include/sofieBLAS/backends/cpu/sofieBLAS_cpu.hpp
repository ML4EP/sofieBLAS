#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>

#if defined(SOFIEBLAS_USE_MKL)
#include <mkl.h>
#elif defined(SOFIEBLAS_USE_OPENBLAS)
#include <cblas.h>
#else
#error                                                                         \
    "No CPU BLAS backend selected. Define SOFIEBLAS_USE_MKL or SOFIEBLAS_USE_OPENBLAS."
#endif

#include <cassert>
#include <cmath>
#include <stdexcept>

class BlasCpu {
public:
  BlasCpu(alpaka::QueueCpuBlocking &queue [[maybe_unused]]) {}

  inline CBLAS_TRANSPOSE charToTranspose(char trans) {
    switch (trans) {
    case 'N':
    case 'n':
      return CblasNoTrans;
    case 'T':
    case 't':
      return CblasTrans;
    case 'C':
    case 'c':
      return CblasConjTrans;
    default:
      throw std::invalid_argument("Invalid transpose character.");
    }
  }

  // C = alpha * op(A) * op(B) + beta * C  (no bias, leading dims inferred)
  template <typename T, typename TIdx>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha,
                     alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const &A,
                     alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const &B,
                     float beta,
                     alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> &C) {
    int lda = (transa == 'N' || transa == 'n') ? static_cast<int>(m)
                                               : static_cast<int>(k);
    int ldb = (transb == 'N' || transb == 'n') ? static_cast<int>(k)
                                               : static_cast<int>(n);
    cblas_sgemm(CblasColMajor, charToTranspose(transa), charToTranspose(transb),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                alpha, alpaka::getPtrNative(A), lda, alpaka::getPtrNative(B),
                ldb, beta, alpaka::getPtrNative(C), static_cast<int>(m));
  }

  // C = alpha * op(A) * op(B) + beta * bias + bias_vec  (bias_vec broadcast per
  // row) Matches the cuBLASLt EPILOGUE_BIAS semantics: the bias buffer serves
  // as both the beta-scaled accumulator and provides the per-row bias vector
  // (first m elements).
  template <typename T, typename TIdx>
  inline void
  gemm(char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
       float alpha, alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const &A,
       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const &B, float beta,
       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> &bias,
       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> &C) {
    int lda = (transa == 'N' || transa == 'n') ? static_cast<int>(m)
                                               : static_cast<int>(k);
    int ldb = (transb == 'N' || transb == 'n') ? static_cast<int>(k)
                                               : static_cast<int>(n);
    // Step 1: C = alpha * op(A) * op(B) (beta=0 so C is fully overwritten)
    cblas_sgemm(CblasColMajor, charToTranspose(transa), charToTranspose(transb),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                alpha, alpaka::getPtrNative(A), lda, alpaka::getPtrNative(B),
                ldb, 0.0f, alpaka::getPtrNative(C), static_cast<int>(m));
    // Step 2: C += beta * bias_matrix + bias_vec (per-row broadcast)
    float *c = alpaka::getPtrNative(C);
    const float *b = alpaka::getPtrNative(bias);
    for (unsigned int j = 0; j < n; ++j)
      for (unsigned int i = 0; i < m; ++i)
        c[j * m + i] += beta * b[j * m + i] + b[i];
  }

  // C = relu(alpha * op(A) * op(B) + beta * bias + bias_vec)
  template <typename T, typename TIdx>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> &C) {
    gemm(transa, transb, m, n, k, alpha, A, B, beta, bias, C);
    float *c = alpaka::getPtrNative(C);
    for (unsigned int i = 0; i < m * n; ++i)
      c[i] = c[i] > 0.0f ? c[i] : 0.0f;
  }

  // C = gelu(alpha * op(A) * op(B) + beta * bias + bias_vec)
  // Uses the standard GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
  template <typename T, typename TIdx>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> &C) {
    gemm(transa, transb, m, n, k, alpha, A, B, beta, bias, C);
    float *c = alpaka::getPtrNative(C);
    constexpr float kInvSqrt2 = 0.7071067811865476f;
    for (unsigned int i = 0; i < m * n; ++i)
      c[i] *= 0.5f * (1.0f + std::erff(c[i] * kInvSqrt2));
  }
};

namespace traits {

template <> class sofieBLAS<alpaka::TagCpuSerial> {
public:
  using Impl = BlasCpu;
};

template <> class sofieBLAS<alpaka::TagCpuOmp2Blocks> {
public:
  using Impl = BlasCpu;
};

template <> class sofieBLAS<alpaka::TagCpuOmp2Threads> {
public:
  using Impl = BlasCpu;
};

template <> class sofieBLAS<alpaka::TagCpuTbbBlocks> {
public:
  using Impl = BlasCpu;
};

template <> class sofieBLAS<alpaka::TagCpuThreads> {
public:
  using Impl = BlasCpu;
};

} // namespace traits

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
