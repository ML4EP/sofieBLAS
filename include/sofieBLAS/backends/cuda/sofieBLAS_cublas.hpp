#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>
#include <cublas_v2.h>

#define CUBLAS_CHECK(err)                                                      \
  if (err != CUBLAS_STATUS_SUCCESS) {                                          \
    std::cerr << "cuBLAS Error at line " << __LINE__ << "\n";                  \
    exit(EXIT_FAILURE);                                                        \
  }

class BlasCuda {
public:
  BlasCuda(alpaka::QueueCudaRtNonBlocking &queue) : m_queue{queue} {
    CUBLAS_CHECK(cublasCreate(&m_handle));
    CUBLAS_CHECK(cublasSetStream(m_handle, m_queue.getNativeHandle()));
  }

  ~BlasCuda() { CUBLAS_CHECK(cublasDestroy(m_handle)); }

  inline cublasOperation_t charToCuBlasTranspose(char trans) {
    switch (trans) {
    case 'N':
    case 'n':
      return CUBLAS_OP_N;
    case 'T':
    case 't':
      return CUBLAS_OP_T;
    case 'C':
    case 'c':
      return CUBLAS_OP_C;
    default:
      throw std::invalid_argument("Invalid transpose character for cuBLAS.");
    }
  }

  template <typename T, typename TIdx>
  inline void
  gemm(char transa, char transb, const unsigned int m, const unsigned int n,
       const unsigned int k, const float alpha,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A, const int lda,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B, const int ldb,
       const float beta, alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C,
       const int ldc) {
    cublasOperation_t opA = charToCuBlasTranspose(transa);
    cublasOperation_t opB = charToCuBlasTranspose(transb);

    CUBLAS_CHECK(cublasSgemm(m_handle, opA, opB, m, n, k, &alpha, A.data(), lda,
                             B.data(), ldb, &beta, C.data(), ldc));
  }

private:
  alpaka::QueueCudaRtNonBlocking m_queue;
  cublasHandle_t m_handle;
};

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuCudaRt> {
public:
  using Impl = BlasCuda;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
