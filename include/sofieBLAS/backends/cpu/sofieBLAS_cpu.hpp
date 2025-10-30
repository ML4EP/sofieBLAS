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

  template <typename T, typename TIdx>
  inline void
  gemm(char transa, char transb, const unsigned int m, const unsigned int n,
       const unsigned int k, const float alpha,
       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const &A, const int lda,
       alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const &B, const int ldb,
       const float beta, alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> &C,
       const int ldc) {
    CBLAS_TRANSPOSE TransA = charToTranspose(transa);
    CBLAS_TRANSPOSE TransB = charToTranspose(transb);
    cblas_sgemm(CblasColMajor, TransA, TransB, m, n, k, alpha, A.data(), lda,
                B.data(), ldb, beta, C.data(), ldc);
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
