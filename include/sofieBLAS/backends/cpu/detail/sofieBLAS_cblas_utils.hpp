#pragma once

// Helpers behind the CPU backends' public API 
// the fp8 / bf16 element types and conversions, the epilogues, 
// and the reference int8 and fp8 GEMMs. Included by 
// sofieBLAS_cblas_common.hpp after the vendor CBLAS header.

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#include "sofieBLAS/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// The element types of the fp8 GEMMs on the CPU: the raw bits of an OCP FP8
// E4M3 number and of a bfloat16 number, with the size and layout of
// the GPU types, so a buffer can be passed to either backend.
struct SofieFp8E4M3 {
  uint8_t bits;
};
struct SofieBf16 {
  uint16_t bits;
};

namespace sofieblas::cpu {

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

// exact conversions between the fp8 / bf16 bits and float
inline float fp8E4M3ToFloat(SofieFp8E4M3 v) {
  const unsigned sign = v.bits >> 7, exp = (v.bits >> 3) & 0xFu,
                 mant = v.bits & 0x7u;
  float x;
  if (exp == 0xFu && mant == 0x7u)
    x = std::numeric_limits<float>::quiet_NaN();
  else if (exp == 0)
    x = std::ldexp(static_cast<float>(mant), -9);
  else
    x = std::ldexp(static_cast<float>(8u + mant), static_cast<int>(exp) - 10);
  return sign ? -x : x;
}

// round to nearest even, saturated to +-448, NaN kept
inline SofieFp8E4M3 floatToFp8E4M3(float x) {
  const uint8_t sign = std::signbit(x) ? 0x80u : 0u;
  if (std::isnan(x))
    return SofieFp8E4M3{static_cast<uint8_t>(sign | 0x7Fu)};
  float a = std::min(std::fabs(x), 448.f);
  if (a < 0.015625f) { // below 2^-6: subnormal, steps of 2^-9
    // 8 steps of 2^-9 is the smallest normal number, whose bits are 0x08
    const auto q = static_cast<uint8_t>(std::nearbyint(std::ldexp(a, 9)));
    return SofieFp8E4M3{static_cast<uint8_t>(sign | q)};
  }
  int e = std::ilogb(a); // a = 1.f * 2^e, e in [-6, 8]
  auto q = static_cast<unsigned>(std::nearbyint(std::ldexp(a, 3 - e)));
  if (q == 16u) { // rounded up to the next power of two
    q = 8u;
    ++e;
  }
  return SofieFp8E4M3{static_cast<uint8_t>(sign | ((e + 7) << 3) | (q - 8u))};
}

inline float bf16ToFloat(SofieBf16 v) {
  const uint32_t bits = static_cast<uint32_t>(v.bits) << 16;
  float x;
  std::memcpy(&x, &bits, sizeof(x));
  return x;
}

inline void checkLeadingDims(char transa, char transb, unsigned int m,
                             unsigned int n, unsigned int k, unsigned int lda,
                             unsigned int ldb) {
  unsigned int rowsA = (transa == 'N' || transa == 'n') ? m : k;
  unsigned int rowsB = (transb == 'N' || transb == 'n') ? k : n;
  if (lda < rowsA || ldb < rowsB)
    throw std::invalid_argument(
        "sofieBLAS: leading dimension smaller than the number of rows of "
        "the matrix.");
}

// GELU, tanh approximation: the definition of the GPU epilogues, so every
// backend computes the same function
inline float geluTanh(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kCubic = 0.044715f;
  return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + kCubic * x * x * x)));
}

// the epilogue of the quantized GEMMs on one value of row i
inline float applyEpilogue(float x, float biasValue, Epilogue epilogue) {
  if (epilogue == Epilogue::Bias || epilogue == Epilogue::ReluBias ||
      epilogue == Epilogue::GeluBias)
    x += biasValue;
  if (epilogue == Epilogue::ReluBias || epilogue == Epilogue::Relu)
    x = x > 0.0f ? x : 0.0f;
  else if (epilogue == Epilogue::GeluBias)
    x = geluTanh(x);
  return x;
}

inline bool epilogueUsesBias(Epilogue epilogue) {
  return epilogue == Epilogue::Bias || epilogue == Epilogue::ReluBias ||
         epilogue == Epilogue::GeluBias;
}

inline void executeInt8Matmul(char transa, char transb, unsigned int m,
                              unsigned int n, unsigned int k, const int8_t *A,
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
inline void executeInt8MatmulMKL(char transa, char transb, unsigned int m,
                                 unsigned int n, unsigned int k,
                                 const int8_t *A, const int8_t *B, int32_t *C) {
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
  cblas_gemm_s8u8s32(CblasColMajor, ta ? CblasTrans : CblasNoTrans,
                     tb ? CblasTrans : CblasNoTrans, CblasFixOffset,
                     static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                     static_cast<MKL_INT>(k), alpha, A, lda, oa, ub.data(), ldb,
                     ob, beta, C, ldc, &oc);
}
#endif

inline void dispatchInt8Matmul(char transa, char transb, unsigned int m,
                               unsigned int n, unsigned int k, const int8_t *A,
                               const int8_t *B, int32_t *C) {
#if defined(SOFIEBLAS_USE_MKL)
  executeInt8MatmulMKL(transa, transb, m, n, k, A, B, C);
#else
  executeInt8Matmul(transa, transb, m, n, k, A, B, C);
#endif
}

// int8 C: float(acc) * alpha (alpha[i] when alphaIsVector), + bias[i], the
// epilogue, round half to even, saturate to [-128, 127]
inline void executeInt8Out(char transa, char transb, unsigned int m,
                           unsigned int n, unsigned int k, const float *alpha,
                           bool alphaIsVector, const int8_t *A, const int8_t *B,
                           const float *bias, int8_t *C, Epilogue epilogue) {
  if (epilogueUsesBias(epilogue) && !bias)
    throw std::invalid_argument("sofieBLAS: bias epilogue without a bias.");
  std::vector<int32_t> acc(static_cast<std::size_t>(m) * n);
  dispatchInt8Matmul(transa, transb, m, n, k, A, B, acc.data());
  for (unsigned int j = 0; j < n; ++j)
    for (unsigned int i = 0; i < m; ++i) {
      const std::size_t idx = static_cast<std::size_t>(j) * m + i;
      const float scaled =
          static_cast<float>(acc[idx]) * (alphaIsVector ? alpha[i] : *alpha);
      float x = applyEpilogue(scaled, bias ? bias[i] : 0.0f, epilogue);
      x = std::nearbyint(x);
      x = std::min(127.0f, std::max(-128.0f, x));
      C[idx] = static_cast<int8_t>(x);
    }
}

// float C of the fp8 GEMMs: decode A and B, multiply in float with alpha *
// scaleA * scaleB, then add the bias and apply the epilogue
inline void executeFp8(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, const SofieFp8E4M3 *A,
                       const float *scaleA, const SofieFp8E4M3 *B,
                       const float *scaleB, const SofieBf16 *bias, float *C,
                       Epilogue epilogue) {
  if (epilogueUsesBias(epilogue) && !bias)
    throw std::invalid_argument("sofieBLAS: bias epilogue without a bias.");
  const std::size_t sizeA = static_cast<std::size_t>(m) * k;
  const std::size_t sizeB = static_cast<std::size_t>(k) * n;
  std::vector<float> fa(sizeA), fb(sizeB);
  for (std::size_t i = 0; i < sizeA; ++i)
    fa[i] = fp8E4M3ToFloat(A[i]);
  for (std::size_t i = 0; i < sizeB; ++i)
    fb[i] = fp8E4M3ToFloat(B[i]);
  const int lda = (transa == 'N' || transa == 'n') ? static_cast<int>(m)
                                                   : static_cast<int>(k);
  const int ldb = (transb == 'N' || transb == 'n') ? static_cast<int>(k)
                                                   : static_cast<int>(n);
  const float scale = alpha * *scaleA * *scaleB;
  cblas_sgemm(CblasColMajor, charToTranspose(transa), charToTranspose(transb),
              static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
              scale, fa.data(), lda, fb.data(), ldb, 0.0f, C,
              static_cast<int>(m));
  if (epilogue == Epilogue::Default)
    return;
  for (unsigned int j = 0; j < n; ++j)
    for (unsigned int i = 0; i < m; ++i) {
      float &x = C[static_cast<std::size_t>(j) * m + i];
      x = applyEpilogue(x, bias ? bf16ToFloat(bias[i]) : 0.0f, epilogue);
    }
}

} // namespace sofieblas::cpu

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
