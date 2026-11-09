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

// The element types of the fp8 and fp16 GEMMs on the CPU: the raw bits of an
// OCP FP8 E4M3 number, of a bfloat16 number and of an IEEE binary16 number,
// with the size and layout of the GPU types, so a buffer can be passed to
// either backend.
struct SofieFp8E4M3 {
  uint8_t bits;
};
struct SofieBf16 {
  uint16_t bits;
};
struct SofieFp16 {
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
inline float fp8E4M3ToFloat(SofieFp8E4M3 encoded) {
  const unsigned signBit = encoded.bits >> 7,
                 exponentBits = (encoded.bits >> 3) & 0xFu,
                 mantissaBits = encoded.bits & 0x7u;
  float value;
  if (exponentBits == 0xFu && mantissaBits == 0x7u)
    value = std::numeric_limits<float>::quiet_NaN();
  else if (exponentBits == 0)
    value = std::ldexp(static_cast<float>(mantissaBits), -9);
  else
    value = std::ldexp(static_cast<float>(8u + mantissaBits),
                       static_cast<int>(exponentBits) - 10);
  return signBit ? -value : value;
}

// round to nearest even, saturated to +-448, NaN kept
inline SofieFp8E4M3 floatToFp8E4M3(float value) {
  const uint8_t signBit = std::signbit(value) ? 0x80u : 0u;
  if (std::isnan(value))
    return SofieFp8E4M3{static_cast<uint8_t>(signBit | 0x7Fu)};
  float magnitude = std::min(std::fabs(value), 448.f);
  if (magnitude < 0.015625f) { // below 2^-6: subnormal, steps of 2^-9
    // 8 steps of 2^-9 is the smallest normal number, whose bits are 0x08
    const auto quantized =
        static_cast<uint8_t>(std::nearbyint(std::ldexp(magnitude, 9)));
    return SofieFp8E4M3{static_cast<uint8_t>(signBit | quantized)};
  }
  int exponent = std::ilogb(magnitude); // a = 1.f * 2^e, e in [-6, 8]
  auto quantized = static_cast<unsigned>(
      std::nearbyint(std::ldexp(magnitude, 3 - exponent)));
  if (quantized == 16u) { // rounded up to the next power of two
    quantized = 8u;
    ++exponent;
  }
  return SofieFp8E4M3{
      static_cast<uint8_t>(signBit | ((exponent + 7) << 3) | (quantized - 8u))};
}

inline float bf16ToFloat(SofieBf16 encoded) {
  const uint32_t bits = static_cast<uint32_t>(encoded.bits) << 16;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// exact conversion of the binary16 bits to float
inline float fp16ToFloat(SofieFp16 encoded) {
  const unsigned signBit = encoded.bits >> 15,
                 exponentBits = (encoded.bits >> 10) & 0x1Fu,
                 mantissaBits = encoded.bits & 0x3FFu;
  float value;
  if (exponentBits == 0x1Fu)
    value = mantissaBits ? std::numeric_limits<float>::quiet_NaN()
                         : std::numeric_limits<float>::infinity();
  else if (exponentBits == 0) // zero and subnormals: steps of 2^-24
    value = std::ldexp(static_cast<float>(mantissaBits), -24);
  else
    value = std::ldexp(static_cast<float>(1024u + mantissaBits),
                       static_cast<int>(exponentBits) - 25);
  return signBit ? -value : value;
}

// round to nearest even; a magnitude of 65520 or more (the midpoint to the
// next power of two above the largest number, 65504) becomes infinity, NaN
// stays NaN
inline SofieFp16 floatToFp16(float value) {
  const uint16_t signBit = std::signbit(value) ? 0x8000u : 0u;
  if (std::isnan(value))
    return SofieFp16{static_cast<uint16_t>(signBit | 0x7E00u)};
  const float magnitude = std::fabs(value);
  if (magnitude >= 65520.f)
    return SofieFp16{static_cast<uint16_t>(signBit | 0x7C00u)};
  if (magnitude < 0.00006103515625f) { // below 2^-14: subnormal, steps of 2^-24
    // 1024 steps of 2^-24 is the smallest normal number, whose bits are 0x0400
    const auto quantized =
        static_cast<uint16_t>(std::nearbyint(std::ldexp(magnitude, 24)));
    return SofieFp16{static_cast<uint16_t>(signBit | quantized)};
  }
  int exponent = std::ilogb(magnitude); // a = 1.f * 2^e, e in [-14, 15]
  auto quantized = static_cast<unsigned>(
      std::nearbyint(std::ldexp(magnitude, 10 - exponent)));
  if (quantized == 2048u) { // rounded up to the next power of two
    quantized = 1024u;
    ++exponent;
  }
  return SofieFp16{static_cast<uint16_t>(signBit | ((exponent + 15) << 10) |
                                         (quantized - 1024u))};
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
inline float geluTanh(float value) {
  constexpr float kSqrt2_OverPi = 0.7978845608028654f;
  constexpr float kCubic = 0.044715f;
  return 0.5f * value *
         (1.0f +
          std::tanh(kSqrt2_OverPi * (value + kCubic * value * value * value)));
}

// the epilogue of the quantized GEMMs on one value of row i
inline float applyEpilogue(float value, float biasVal, Epilogue epilogue) {
  if (epilogue == Epilogue::Bias || epilogue == Epilogue::ReluBias ||
      epilogue == Epilogue::GeluBias)
    value += biasVal;
  if (epilogue == Epilogue::ReluBias || epilogue == Epilogue::Relu)
    value = value > 0.0f ? value : 0.0f;
  else if (epilogue == Epilogue::GeluBias || epilogue == Epilogue::Gelu)
    value = geluTanh(value);
  return value;
}

inline bool epilogueUsesBias(Epilogue epilogue) {
  return epilogue == Epilogue::Bias || epilogue == Epilogue::ReluBias ||
         epilogue == Epilogue::GeluBias;
}

inline void executeInt8Matmul(char transa, char transb, unsigned int m,
                              unsigned int n, unsigned int k, const int8_t *A,
                              const int8_t *B, int32_t *C) {
  bool transposeA = (transa == 'T' || transa == 't');
  bool transposeB = (transb == 'T' || transb == 't');
  int lda = transposeA ? static_cast<int>(k) : static_cast<int>(m);
  int ldb = transposeB ? static_cast<int>(n) : static_cast<int>(k);
  auto elementAt = [](const int8_t *matrix, int row, int col,
                      int leadingDimension) -> int32_t {
    return matrix[col * leadingDimension + row];
  };
  for (unsigned int j = 0; j < n; ++j) {
    for (unsigned int i = 0; i < m; ++i) {
      int32_t dotProduct = 0;
      for (unsigned int p = 0; p < k; ++p) {
        int32_t valueA =
            transposeA
                ? elementAt(A, static_cast<int>(p), static_cast<int>(i), lda)
                : elementAt(A, static_cast<int>(i), static_cast<int>(p), lda);
        int32_t valueB =
            transposeB
                ? elementAt(B, static_cast<int>(j), static_cast<int>(p), ldb)
                : elementAt(B, static_cast<int>(p), static_cast<int>(j), ldb);
        dotProduct += valueA * valueB;
      }
      C[j * m + i] = dotProduct;
    }
  }
}

#if defined(SOFIEBLAS_USE_MKL)
inline void executeInt8MatmulMKL(char transa, char transb, unsigned int m,
                                 unsigned int n, unsigned int k,
                                 const int8_t *A, const int8_t *B, int32_t *C) {
  bool transposeA = (transa == 'T' || transa == 't');
  bool transposeB = (transb == 'T' || transb == 't');
  int lda = transposeA ? static_cast<int>(k) : static_cast<int>(m);
  int ldb = transposeB ? static_cast<int>(n) : static_cast<int>(k);
  int ldc = static_cast<int>(m);

  // B is stored as k*n contiguous int8 elements regardless of transb (the
  // transpose flag only changes how those elements are interpreted).
  std::vector<uint8_t> unsignedB(static_cast<std::size_t>(k) *
                                 static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < unsignedB.size(); ++i)
    unsignedB[i] = static_cast<uint8_t>(B[i]) ^ 0x80u;

  const float alpha = 1.0f, beta = 0.0f;
  const MKL_INT8 offsetA = 0, offsetB = -128;
  const MKL_INT32 offsetC = 0;
  cblas_gemm_s8u8s32(CblasColMajor, transposeA ? CblasTrans : CblasNoTrans,
                     transposeB ? CblasTrans : CblasNoTrans, CblasFixOffset,
                     static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                     static_cast<MKL_INT>(k), alpha, A, lda, offsetA,
                     unsignedB.data(), ldb, offsetB, beta, C, ldc, &offsetC);
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
  std::vector<int32_t> accumulators(static_cast<std::size_t>(m) * n);
  dispatchInt8Matmul(transa, transb, m, n, k, A, B, accumulators.data());
  for (unsigned int j = 0; j < n; ++j)
    for (unsigned int i = 0; i < m; ++i) {
      const std::size_t flatIdx = static_cast<std::size_t>(j) * m + i;
      const float scaled = static_cast<float>(accumulators[flatIdx]) *
                           (alphaIsVector ? alpha[i] : *alpha);
      float value = applyEpilogue(scaled, bias ? bias[i] : 0.0f, epilogue);
      value = std::nearbyint(value);
      value = std::min(127.0f, std::max(-128.0f, value));
      C[flatIdx] = static_cast<int8_t>(value);
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
  std::vector<float> floatA(sizeA), floatB(sizeB);
  for (std::size_t i = 0; i < sizeA; ++i)
    floatA[i] = fp8E4M3ToFloat(A[i]);
  for (std::size_t i = 0; i < sizeB; ++i)
    floatB[i] = fp8E4M3ToFloat(B[i]);
  const int lda = (transa == 'N' || transa == 'n') ? static_cast<int>(m)
                                                   : static_cast<int>(k);
  const int ldb = (transb == 'N' || transb == 'n') ? static_cast<int>(k)
                                                   : static_cast<int>(n);
  const float scale = alpha * *scaleA * *scaleB;
  cblas_sgemm(CblasColMajor, charToTranspose(transa), charToTranspose(transb),
              static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
              scale, floatA.data(), lda, floatB.data(), ldb, 0.0f, C,
              static_cast<int>(m));
  if (epilogue == Epilogue::Default)
    return;
  for (unsigned int j = 0; j < n; ++j)
    for (unsigned int i = 0; i < m; ++i) {
      float &value = C[static_cast<std::size_t>(j) * m + i];
      value =
          applyEpilogue(value, bias ? bf16ToFloat(bias[i]) : 0.0f, epilogue);
    }
}

// float C of the fp16 GEMMs: decode A and B, then
// C = epilogue(alpha * op(A) * op(B) + beta * C + bias) with the sum, the scale
// and the epilogue in float. C holds the input of the beta term and the
// result; bias (one value per row of C) may be null when the epilogue adds
// none.
inline void executeFp16(char transa, char transb, unsigned int m,
                        unsigned int n, unsigned int k, float alpha,
                        const SofieFp16 *A, const SofieFp16 *B, float beta,
                        const float *bias, float *C, Epilogue epilogue) {
  if (epilogueUsesBias(epilogue) && !bias)
    throw std::invalid_argument("sofieBLAS: bias epilogue without a bias.");
  const std::size_t sizeA = static_cast<std::size_t>(m) * k;
  const std::size_t sizeB = static_cast<std::size_t>(k) * n;
  std::vector<float> floatA(sizeA), floatB(sizeB);
  for (std::size_t i = 0; i < sizeA; ++i)
    floatA[i] = fp16ToFloat(A[i]);
  for (std::size_t i = 0; i < sizeB; ++i)
    floatB[i] = fp16ToFloat(B[i]);
  const int lda = (transa == 'N' || transa == 'n') ? static_cast<int>(m)
                                                   : static_cast<int>(k);
  const int ldb = (transb == 'N' || transb == 'n') ? static_cast<int>(k)
                                                   : static_cast<int>(n);
  cblas_sgemm(CblasColMajor, charToTranspose(transa), charToTranspose(transb),
              static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
              alpha, floatA.data(), lda, floatB.data(), ldb, beta, C,
              static_cast<int>(m));
  if (epilogue == Epilogue::Default)
    return;
  for (unsigned int j = 0; j < n; ++j)
    for (unsigned int i = 0; i < m; ++i) {
      float &value = C[static_cast<std::size_t>(j) * m + i];
      value = applyEpilogue(value, bias ? bias[i] : 0.0f, epilogue);
    }
}

} // namespace sofieblas::cpu

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
