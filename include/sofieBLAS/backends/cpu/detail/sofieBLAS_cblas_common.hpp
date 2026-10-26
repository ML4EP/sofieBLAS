#pragma once

// Shared implementation for every CPU backend that exposes a standard CBLAS
// API. The vendor-specific header (cblas.h / mkl.h / blis/cblas.h /
// Accelerate.h) must already be included by the backend wrapper before this
// file, since they all provide the same CBLAS symbols.

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#include "sofieBLAS/backends/cpu/detail/sofieBLAS_cblas_utils.hpp"
#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

class BlasCpu {
public:
  // the bf16 and fp8 element types of this backend, for the fp8 GEMMs
  using Bf16 = SofieBf16;
  using Fp8E4M3 = SofieFp8E4M3;
  using Fp8E4M3Type = SofieFp8E4M3;
  using Bf16Type = SofieBf16;

  BlasCpu(alpaka::QueueCpuBlocking &queue [[maybe_unused]]) {}

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
    cblas_sgemm(CblasColMajor, sofieblas::cpu::charToTranspose(transa),
                sofieblas::cpu::charToTranspose(transb), static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha,
                alpaka::getPtrNative(A), lda, alpaka::getPtrNative(B), ldb,
                beta, alpaka::getPtrNative(C), static_cast<int>(m));
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
    cblas_sgemm(CblasColMajor, sofieblas::cpu::charToTranspose(transa),
                sofieblas::cpu::charToTranspose(transb), static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha,
                alpaka::getPtrNative(A), lda, alpaka::getPtrNative(B), ldb,
                beta, alpaka::getPtrNative(C), static_cast<int>(m));
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
    cblas_sgemm(CblasColMajor, sofieblas::cpu::charToTranspose(transa),
                sofieblas::cpu::charToTranspose(transb), static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha,
                alpaka::getPtrNative(A), lda, alpaka::getPtrNative(B), ldb,
                0.0f, alpaka::getPtrNative(C), static_cast<int>(m));
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
    cblas_sgemm(CblasColMajor, sofieblas::cpu::charToTranspose(transa),
                sofieblas::cpu::charToTranspose(transb), static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha,
                alpaka::getPtrNative(A), lda, alpaka::getPtrNative(B), ldb,
                0.0f, alpaka::getPtrNative(C), static_cast<int>(m));
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
  // Uses the tanh approximation of GELU, as the GPU epilogues do
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
    for (unsigned int i = 0; i < m * n; ++i)
      c[i] = sofieblas::cpu::geluTanh(c[i]);
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
    for (unsigned int i = 0; i < m * n; ++i)
      c[i] = sofieblas::cpu::geluTanh(c[i]);
  }

  template <typename TA, typename TB, typename TC>
  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k, TA const &A,
                         TB const &B, TC &C) {
    sofieblas::cpu::dispatchInt8Matmul(
        transa, transb, m, n, k, alpaka::getPtrNative(A),
        alpaka::getPtrNative(B), alpaka::getPtrNative(C));
  }

  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k, const int8_t *A,
                         const int8_t *B, int32_t *C) {
    sofieblas::cpu::dispatchInt8Matmul(transa, transb, m, n, k, A, B, C);
  }

  // int8 GEMMs with an int8 C, as on the GPU backends: the exact int32 sum is
  // scaled in float by alpha, the bias (one value per row of C) is added, the
  // relu epilogues clamp at 0, and the result is rounded to the nearest
  // integer (ties to even) and saturated to [-128, 127]. Pointers are host
  // pointers; bias may be null when the epilogue adds none.
  template <typename TA, typename TB, typename TBias, typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k, float alpha,
                         TA const &A, TB const &B, TBias &bias, TC &C,
                         Epilogue epilogue) {
    int8Matmul(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
               alpaka::getPtrNative(B), alpaka::getPtrNative(bias),
               alpaka::getPtrNative(C), epilogue);
  }

  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k, float alpha,
                         const int8_t *A, const int8_t *B, const float *bias,
                         int8_t *C, Epilogue epilogue) {
    sofieblas::cpu::executeInt8Out(transa, transb, m, n, k, &alpha, false, A, B,
                                   bias, C, epilogue);
  }

  // as the int8-output int8Matmul, with one alpha per row of C (alphaVector
  // holds m floats)
  template <typename TAlpha, typename TA, typename TB, typename TBias,
            typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void int8MatmulVectorAlpha(char transa, char transb, unsigned int m,
                                    unsigned int n, unsigned int k,
                                    TAlpha const &alphaVector, TA const &A,
                                    TB const &B, TBias &bias, TC &C,
                                    Epilogue epilogue) {
    int8MatmulVectorAlpha(transa, transb, m, n, k,
                          alpaka::getPtrNative(alphaVector),
                          alpaka::getPtrNative(A), alpaka::getPtrNative(B),
                          alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                          epilogue);
  }

  inline void int8MatmulVectorAlpha(char transa, char transb, unsigned int m,
                                    unsigned int n, unsigned int k,
                                    const float *alphaVector, const int8_t *A,
                                    const int8_t *B, const float *bias,
                                    int8_t *C, Epilogue epilogue) {
    sofieblas::cpu::executeInt8Out(transa, transb, m, n, k, alphaVector, true,
                                   A, B, bias, C, epilogue);
  }

  // Strided-batched int8 GEMMs: C_b = op(A_b) * op(B_b), X_b = X + b * strideX
  // (strides in elements, dense matrices, 0 shares the matrix). The first
  // writes the int32 sum, the second an int8 C as the int8-output int8Matmul
  // with a bias shared by the batches.
  template <typename TA, typename TB, typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void int8MatmulStridedBatched(char transa, char transb,
                                       unsigned int m, unsigned int n,
                                       unsigned int k, TA const &A,
                                       long long strideA, TB const &B,
                                       long long strideB, TC &C,
                                       long long strideC, int batchCount) {
    int8MatmulStridedBatched(transa, transb, m, n, k, alpaka::getPtrNative(A),
                             strideA, alpaka::getPtrNative(B), strideB,
                             alpaka::getPtrNative(C), strideC, batchCount);
  }

  inline void int8MatmulStridedBatched(char transa, char transb,
                                       unsigned int m, unsigned int n,
                                       unsigned int k, const int8_t *A,
                                       long long strideA, const int8_t *B,
                                       long long strideB, int32_t *C,
                                       long long strideC, int batchCount) {
    if (batchCount < 1)
      throw std::invalid_argument("sofieBLAS: batchCount must be positive.");
    for (int b = 0; b < batchCount; ++b)
      sofieblas::cpu::dispatchInt8Matmul(transa, transb, m, n, k,
                                         A + b * strideA, B + b * strideB,
                                         C + b * strideC);
  }

  template <typename TA, typename TB, typename TBias, typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void int8MatmulStridedBatched(char transa, char transb,
                                       unsigned int m, unsigned int n,
                                       unsigned int k, float alpha,
                                       TA const &A, long long strideA,
                                       TB const &B, long long strideB,
                                       TBias &bias, TC &C, long long strideC,
                                       int batchCount, Epilogue epilogue) {
    int8MatmulStridedBatched(transa, transb, m, n, k, alpha,
                             alpaka::getPtrNative(A), strideA,
                             alpaka::getPtrNative(B), strideB,
                             alpaka::getPtrNative(bias),
                             alpaka::getPtrNative(C), strideC, batchCount,
                             epilogue);
  }

  inline void int8MatmulStridedBatched(char transa, char transb,
                                       unsigned int m, unsigned int n,
                                       unsigned int k, float alpha,
                                       const int8_t *A, long long strideA,
                                       const int8_t *B, long long strideB,
                                       const float *bias, int8_t *C,
                                       long long strideC, int batchCount,
                                       Epilogue epilogue) {
    if (batchCount < 1)
      throw std::invalid_argument("sofieBLAS: batchCount must be positive.");
    for (int b = 0; b < batchCount; ++b)
      sofieblas::cpu::executeInt8Out(transa, transb, m, n, k, &alpha, false,
                                     A + b * strideA, B + b * strideB, bias,
                                     C + b * strideC, epilogue);
  }

  // fp8 GEMMs, as on the GPU backends: A and B in FP8 E4M3, each with its own
  // scale (a host pointer to one float), summed in float:
  // C = epilogue(alpha * scaleA * scaleB * op(A) * op(B) + bias), with a BF16
  // bias of one value per row of C. matmulFp8Fp32 writes a float C;
  // matmulFp8Fp8 multiplies by scaleD and writes E4M3 (round to nearest even,
  // saturated to +-448, NaN kept). The GELU epilogue is the tanh one.
  template <typename TA, typename TSA, typename TB, typename TSB,
            typename TBias, typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void matmulFp8Fp32(char transa, char transb, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            TA const &A, TSA const &scaleA, TB const &B,
                            TSB const &scaleB, TBias &bias, TC &C,
                            Epilogue epilogue = Epilogue::Default) {
    matmulFp8Fp32(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(scaleA), alpaka::getPtrNative(B),
                  alpaka::getPtrNative(scaleB), alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C), epilogue);
  }

  inline void matmulFp8Fp32(char transa, char transb, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            const Fp8E4M3 *A, const float *scaleA,
                            const Fp8E4M3 *B, const float *scaleB,
                            const Bf16 *bias, float *C,
                            Epilogue epilogue = Epilogue::Default) {
    sofieblas::cpu::executeFp8(transa, transb, m, n, k, alpha, A, scaleA, B,
                               scaleB, bias, C, epilogue);
  }

  template <typename TA, typename TSA, typename TB, typename TSB,
            typename TBias, typename TC, typename TSD,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void matmulFp8Fp8(char transa, char transb, unsigned int m,
                           unsigned int n, unsigned int k, float alpha,
                           TA const &A, TSA const &scaleA, TB const &B,
                           TSB const &scaleB, TBias &bias, TC &C,
                           TSD const &scaleD,
                           Epilogue epilogue = Epilogue::Default) {
    matmulFp8Fp8(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
                 alpaka::getPtrNative(scaleA), alpaka::getPtrNative(B),
                 alpaka::getPtrNative(scaleB), alpaka::getPtrNative(bias),
                 alpaka::getPtrNative(C), alpaka::getPtrNative(scaleD),
                 epilogue);
  }

  inline void matmulFp8Fp8(char transa, char transb, unsigned int m,
                           unsigned int n, unsigned int k, float alpha,
                           const Fp8E4M3 *A, const float *scaleA,
                           const Fp8E4M3 *B, const float *scaleB,
                           const Bf16 *bias, Fp8E4M3 *C, const float *scaleD,
                           Epilogue epilogue = Epilogue::Default) {
    std::vector<float> out(static_cast<std::size_t>(m) * n);
    sofieblas::cpu::executeFp8(transa, transb, m, n, k, alpha, A, scaleA, B,
                               scaleB, bias, out.data(), epilogue);
    const float sD = *scaleD;
    for (std::size_t i = 0; i < out.size(); ++i)
      C[i] = sofieblas::cpu::floatToFp8E4M3(sD * out[i]);
  }

  // Strided-batched matmulFp8Fp32: X_b = X + b * strideX (strides in elements,
  // dense matrices, 0 shares the matrix); scales and bias shared by the
  // batches
  template <typename TA, typename TSA, typename TB, typename TSB,
            typename TBias, typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void matmulFp8Fp32StridedBatched(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha, TA const &A, TSA const &scaleA, long long strideA,
      TB const &B, TSB const &scaleB, long long strideB, TBias &bias, TC &C,
      long long strideC, int batchCount,
      Epilogue epilogue = Epilogue::Default) {
    matmulFp8Fp32StridedBatched(
        transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
        alpaka::getPtrNative(scaleA), strideA, alpaka::getPtrNative(B),
        alpaka::getPtrNative(scaleB), strideB, alpaka::getPtrNative(bias),
        alpaka::getPtrNative(C), strideC, batchCount, epilogue);
  }

  inline void matmulFp8Fp32StridedBatched(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha, const Fp8E4M3 *A, const float *scaleA, long long strideA,
      const Fp8E4M3 *B, const float *scaleB, long long strideB,
      const Bf16 *bias, float *C, long long strideC, int batchCount,
      Epilogue epilogue = Epilogue::Default) {
    if (batchCount < 1)
      throw std::invalid_argument("sofieBLAS: batchCount must be positive.");
    for (int b = 0; b < batchCount; ++b)
      sofieblas::cpu::executeFp8(transa, transb, m, n, k, alpha,
                                 A + b * strideA, scaleA, B + b * strideB,
                                 scaleB, bias, C + b * strideC, epilogue);
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
    cblas_sgemm(CblasColMajor, sofieblas::cpu::charToTranspose(transa),
                sofieblas::cpu::charToTranspose(transb), static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha, A, lda, B, ldb,
                beta, C, static_cast<int>(m));
  }

  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, T const *B,
                   float beta, T *bias, T *C) {
    int lda = (transa == 'N' || transa == 'n') ? static_cast<int>(m)
                                               : static_cast<int>(k);
    int ldb = (transb == 'N' || transb == 'n') ? static_cast<int>(k)
                                               : static_cast<int>(n);
    cblas_sgemm(CblasColMajor, sofieblas::cpu::charToTranspose(transa),
                sofieblas::cpu::charToTranspose(transb), static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha, A, lda, B, ldb,
                0.0f, C, static_cast<int>(m));
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
    for (unsigned int i = 0; i < m * n; ++i)
      C[i] = sofieblas::cpu::geluTanh(C[i]);
  }

  // Variants taking the leading dimensions lda and ldb of A and B (column-major
  // physical rows: lda >= m for 'N', >= k for 'T'; ldb >= k for 'N', >= n for
  // 'T'). C is dense.
  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, unsigned int lda,
                     T const *B, unsigned int ldb, float beta, T *C) {
    sofieblas::cpu::checkLeadingDims(transa, transb, m, n, k, lda, ldb);
    cblas_sgemm(CblasColMajor, sofieblas::cpu::charToTranspose(transa),
                sofieblas::cpu::charToTranspose(transb), static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha, A,
                static_cast<int>(lda), B, static_cast<int>(ldb), beta, C,
                static_cast<int>(m));
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
    for (unsigned int i = 0; i < m * n; ++i)
      C[i] = sofieblas::cpu::geluTanh(C[i]);
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
    sofieblas::cpu::checkLeadingDims(transa, transb, m, n, k, lda, ldb);
    if (ldc < m)
      throw std::invalid_argument(
          "sofieBLAS: leading dimension smaller than the number of rows of "
          "the matrix.");
    for (int b = 0; b < batchCount; ++b) {
      float *c = C + b * strideC;
      cblas_sgemm(CblasColMajor, sofieblas::cpu::charToTranspose(transa),
                  sofieblas::cpu::charToTranspose(transb), m, n, k, alpha,
                  A + b * strideA, lda, B + b * strideB, ldb, beta, c, ldc);
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
            x = sofieblas::cpu::geluTanh(x);
        }
    }
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
