#pragma once

// Shared implementation for every CPU backend that exposes a standard CBLAS
// API. The vendor-specific header (cblas.h / mkl.h / blis/cblas.h /
// Accelerate.h) must already be included by the backend wrapper before this
// file, since they all provide the same CBLAS symbols.

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

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

  template <typename T, typename TIdx>
  inline void
  matmul(char transa, char transb, unsigned int m, unsigned int n,
         unsigned int k, float alpha,
         alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> const
             &A,
         alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> const
             &B,
         float beta,
         alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> &C) {
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
  // row)
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
    cblas_sgemm(CblasColMajor, charToTranspose(transa), charToTranspose(transb),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                alpha, alpaka::getPtrNative(A), lda, alpaka::getPtrNative(B),
                ldb, 0.0f, alpaka::getPtrNative(C), static_cast<int>(m));
    float *c = alpaka::getPtrNative(C);
    const float *b = alpaka::getPtrNative(bias);
    for (unsigned int j = 0; j < n; ++j)
      for (unsigned int i = 0; i < m; ++i)
        c[j * m + i] += beta * b[j * m + i] + b[i];
  }

  template <typename T, typename TIdx>
  inline void
  gemm(char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
       float alpha,
       alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> const
           &A,
       alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> const
           &B,
       float beta,
       alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> &bias,
       alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> &C) {
    int lda = (transa == 'N' || transa == 'n') ? static_cast<int>(m)
                                               : static_cast<int>(k);
    int ldb = (transb == 'N' || transb == 'n') ? static_cast<int>(k)
                                               : static_cast<int>(n);
    cblas_sgemm(CblasColMajor, charToTranspose(transa), charToTranspose(transb),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                alpha, alpaka::getPtrNative(A), lda, alpaka::getPtrNative(B),
                ldb, 0.0f, alpaka::getPtrNative(C), static_cast<int>(m));
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

  template <typename T, typename TIdx>
  inline void gemmrelu(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha,
      alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> const
          &A,
      alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> const
          &B,
      float beta,
      alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> &bias,
      alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> &C) {
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

  template <typename T, typename TIdx>
  inline void gemmgelu(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha,
      alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> const
          &A,
      alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> const
          &B,
      float beta,
      alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> &bias,
      alpaka::ViewPlainPtr<alpaka::DevCpu, T, alpaka::DimInt<1u>, TIdx> &C) {
    gemm(transa, transb, m, n, k, alpha, A, B, beta, bias, C);
    float *c = alpaka::getPtrNative(C);
    constexpr float kInvSqrt2 = 0.7071067811865476f;
    for (unsigned int i = 0; i < m * n; ++i)
      c[i] *= 0.5f * (1.0f + std::erff(c[i] * kInvSqrt2));
  }

  template <typename TA, typename TB, typename TC>
  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k, TA const &A,
                         TB const &B, TC &C) {
    dispatchInt8Matmul(transa, transb, m, n, k, alpaka::getPtrNative(A),
                      alpaka::getPtrNative(B), alpaka::getPtrNative(C));
  }

  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k, const int8_t *A,
                         const int8_t *B, int32_t *C) {
    dispatchInt8Matmul(transa, transb, m, n, k, A, B, C);
  }

  // Raw-pointer overloads: accept T const*/T* from any BufXxx or ViewPlainPtr
  // via getPtrNative()
  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, T const *B,
                     float beta, T *C) {
    int lda = (transa == 'N' || transa == 'n') ? static_cast<int>(m)
                                               : static_cast<int>(k);
    int ldb = (transb == 'N' || transb == 'n') ? static_cast<int>(k)
                                               : static_cast<int>(n);
    cblas_sgemm(CblasColMajor, charToTranspose(transa), charToTranspose(transb),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                alpha, A, lda, B, ldb, beta, C, static_cast<int>(m));
  }

  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, T const *B,
                   float beta, T *bias, T *C) {
    int lda = (transa == 'N' || transa == 'n') ? static_cast<int>(m)
                                               : static_cast<int>(k);
    int ldb = (transb == 'N' || transb == 'n') ? static_cast<int>(k)
                                               : static_cast<int>(n);
    cblas_sgemm(CblasColMajor, charToTranspose(transa), charToTranspose(transb),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                alpha, A, lda, B, ldb, 0.0f, C, static_cast<int>(m));
    for (unsigned int j = 0; j < n; ++j)
      for (unsigned int i = 0; i < m; ++i)
        C[j * m + i] += beta * bias[j * m + i] + bias[i];
  }

  template <typename T>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    gemm(transa, transb, m, n, k, alpha, A, B, beta, bias, C);
    for (unsigned int i = 0; i < m * n; ++i)
      C[i] = C[i] > 0.0f ? C[i] : 0.0f;
  }

  template <typename T>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    gemm(transa, transb, m, n, k, alpha, A, B, beta, bias, C);
    constexpr float kInvSqrt2 = 0.7071067811865476f;
    for (unsigned int i = 0; i < m * n; ++i)
      C[i] *= 0.5f * (1.0f + std::erff(C[i] * kInvSqrt2));
  }

  // Variants taking the leading dimensions lda and ldb of A and B (column-major
  // physical rows: lda >= m for 'N', >= k for 'T'; ldb >= k for 'N', >= n for
  // 'T'). C is dense.
  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, unsigned int lda,
                     T const *B, unsigned int ldb, float beta, T *C) {
    checkLeadingDims(transa, transb, m, n, k, lda, ldb);
    cblas_sgemm(CblasColMajor, charToTranspose(transa), charToTranspose(transb),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                alpha, A, static_cast<int>(lda), B, static_cast<int>(ldb), beta,
                C, static_cast<int>(m));
  }

  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, unsigned int lda,
                   T const *B, unsigned int ldb, float beta, T *bias, T *C) {
    matmul(transa, transb, m, n, k, alpha, A, lda, B, ldb, 0.0f, C);
    for (unsigned int j = 0; j < n; ++j)
      for (unsigned int i = 0; i < m; ++i)
        C[j * m + i] += beta * bias[j * m + i] + bias[i];
  }

  template <typename T>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A,
                       unsigned int lda, T const *B, unsigned int ldb,
                       float beta, T *bias, T *C) {
    gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, bias, C);
    for (unsigned int i = 0; i < m * n; ++i)
      C[i] = C[i] > 0.0f ? C[i] : 0.0f;
  }

  template <typename T>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A,
                       unsigned int lda, T const *B, unsigned int ldb,
                       float beta, T *bias, T *C) {
    gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, bias, C);
    constexpr float kInvSqrt2 = 0.7071067811865476f;
    for (unsigned int i = 0; i < m * n; ++i)
      C[i] *= 0.5f * (1.0f + std::erff(C[i] * kInvSqrt2));
  }

  // Batched multiply with a fused epilogue
  // C_b = epilogue(alpha * op(A_b) * op(B_b) + beta * C_b + bias_b)
  // with X_b = X + b * strideX (strides in elements, 0 shares the matrix between
  // the batches) and bias_b = bias + b * strideBias, a vector with one element
  // per row of C.
  inline void gemmStridedBatched(char transa, char transb, int m, int n, int k,
                                 float alpha, const float *A, int lda,
                                 long long strideA, const float *B, int ldb,
                                 long long strideB, float beta, float *C,
                                 int ldc, long long strideC, int batchCount,
                                 Epilogue epilogue, const float *bias,
                                 long long strideBias = 0) {
    if (batchCount < 1)
      throw std::invalid_argument("sofieBLAS: batchCount must be positive.");
    checkLeadingDims(transa, transb, m, n, k, lda, ldb);
    if (ldc < m)
      throw std::invalid_argument(
          "sofieBLAS: leading dimension smaller than the number of rows of "
          "the matrix.");
    constexpr float kInvSqrt2 = 0.7071067811865476f;
    for (int b = 0; b < batchCount; ++b) {
      float *c = C + b * strideC;
      cblas_sgemm(CblasColMajor, charToTranspose(transa),
                  charToTranspose(transb), m, n, k, alpha, A + b * strideA,
                  lda, B + b * strideB, ldb, beta, c, ldc);
      if (epilogue == Epilogue::Default)
        continue;
      const float *bv = bias + b * strideBias;
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
          float &x = c[j * ldc + i];
          x += bv[i];
          if (epilogue == Epilogue::ReluBias)
            x = x > 0.0f ? x : 0.0f;
          else if (epilogue == Epilogue::GeluBias)
            x *= 0.5f * (1.0f + std::erff(x * kInvSqrt2));
        }
    }
  }

private:
  static inline void checkLeadingDims(char transa, char transb, unsigned int m,
                                      unsigned int n, unsigned int k,
                                      unsigned int lda, unsigned int ldb) {
    unsigned int rowsA = (transa == 'N' || transa == 'n') ? m : k;
    unsigned int rowsB = (transb == 'N' || transb == 'n') ? k : n;
    if (lda < rowsA || ldb < rowsB)
      throw std::invalid_argument(
          "sofieBLAS: leading dimension smaller than the number of rows of "
          "the matrix.");
  }

  static inline void executeInt8Matmul(char transa, char transb,
                                       unsigned int m, unsigned int n,
                                       unsigned int k, const int8_t *A,
                                       const int8_t *B, int32_t *C) {
    bool ta = (transa == 'T' || transa == 't');
    bool tb = (transb == 'T' || transb == 't');
    int lda = ta ? static_cast<int>(k) : static_cast<int>(m);
    int ldb = tb ? static_cast<int>(n) : static_cast<int>(k);
    auto at = [](const int8_t *M, int row, int col, int ld) -> int32_t {
      return M[col * ld + row];
    };
    for (unsigned int j = 0; j < n; ++j) {
      for (unsigned int i = 0; i < m; ++i) {
        int32_t sum = 0;
        for (unsigned int p = 0; p < k; ++p) {
          int32_t a = ta ? at(A, static_cast<int>(p), static_cast<int>(i), lda)
                         : at(A, static_cast<int>(i), static_cast<int>(p), lda);
          int32_t b = tb ? at(B, static_cast<int>(j), static_cast<int>(p), ldb)
                         : at(B, static_cast<int>(p), static_cast<int>(j), ldb);
          sum += a * b;
        }
        C[j * m + i] = sum;
      }
    }
  }

#if defined(SOFIEBLAS_USE_MKL)
  static inline void executeInt8MatmulMKL(char transa, char transb,
                                          unsigned int m, unsigned int n,
                                          unsigned int k, const int8_t *A,
                                          const int8_t *B, int32_t *C) {
    bool ta = (transa == 'T' || transa == 't');
    bool tb = (transb == 'T' || transb == 't');
    int lda = ta ? static_cast<int>(k) : static_cast<int>(m);
    int ldb = tb ? static_cast<int>(n) : static_cast<int>(k);
    int ldc = static_cast<int>(m);

    // B is stored as k*n contiguous int8 elements regardless of transb (the
    // transpose flag only changes how those elements are interpreted).
    std::vector<uint8_t> ub(static_cast<std::size_t>(k) *
                            static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < ub.size(); ++i)
      ub[i] = static_cast<uint8_t>(B[i]) ^ 0x80u;

    const float alpha = 1.0f, beta = 0.0f;
    const MKL_INT8 oa = 0, ob = -128;
    const MKL_INT32 oc = 0;
    cblas_gemm_s8u8s32(
        CblasColMajor, ta ? CblasTrans : CblasNoTrans,
        tb ? CblasTrans : CblasNoTrans, CblasFixOffset,
        static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
        static_cast<MKL_INT>(k), alpha, A, lda, oa, ub.data(), ldb, ob, beta,
        C, ldc, &oc);
  }
#endif

  static inline void dispatchInt8Matmul(char transa, char transb,
                                        unsigned int m, unsigned int n,
                                        unsigned int k, const int8_t *A,
                                        const int8_t *B, int32_t *C) {
#if defined(SOFIEBLAS_USE_MKL)
    executeInt8MatmulMKL(transa, transb, m, n, k, A, B, C);
#else
    executeInt8Matmul(transa, transb, m, n, k, A, B, C);
#endif
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
