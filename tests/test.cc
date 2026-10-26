#include "sofieBLAS/sofieBLAS.hpp"
#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Idx = uint32_t;
using Dim1D = alpaka::DimInt<1u>;

// ---------------------------------------------------------------------------
// Reference implementations (column-major, float)
// ---------------------------------------------------------------------------
static inline float cm(const float *M, int row, int col, int ld) {
  return M[col * ld + row];
}

// C = alpha * op(A) * op(B) + beta * C  (in-place, column-major)
static void refMatmul(float *C, const float *A, const float *B, int m, int n,
                      int k, float alpha, float beta, bool transA,
                      bool transB) {
  int lda = transA ? k : m;
  int ldb = transB ? n : k;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      float sum = 0.f;
      for (int p = 0; p < k; ++p) {
        float a = transA ? cm(A, p, i, lda) : cm(A, i, p, lda);
        float b = transB ? cm(B, j, p, ldb) : cm(B, p, j, ldb);
        sum += a * b;
      }
      C[j * m + i] = alpha * sum + beta * C[j * m + i];
    }
  }
}

// C = alpha * op(A) * op(B) + beta * bias_matrix + bias_vec (per-row broadcast)
static void refGemm(float *C, const float *A, const float *B, const float *bias,
                    int m, int n, int k, float alpha, float beta, bool transA,
                    bool transB) {
  int lda = transA ? k : m;
  int ldb = transB ? n : k;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      float sum = 0.f;
      for (int p = 0; p < k; ++p) {
        float a = transA ? cm(A, p, i, lda) : cm(A, i, p, lda);
        float b = transB ? cm(B, j, p, ldb) : cm(B, p, j, ldb);
        sum += a * b;
      }
      C[j * m + i] = alpha * sum + beta * bias[j * m + i] + bias[i];
    }
  }
}

static void refGemmRelu(float *C, const float *A, const float *B,
                        const float *bias, int m, int n, int k, float alpha,
                        float beta, bool transA, bool transB) {
  refGemm(C, A, B, bias, m, n, k, alpha, beta, transA, transB);
  for (int i = 0; i < m * n; ++i)
    C[i] = C[i] > 0.f ? C[i] : 0.f;
}

static void refGemmGelu(float *C, const float *A, const float *B,
                        const float *bias, int m, int n, int k, float alpha,
                        float beta, bool transA, bool transB) {
  refGemm(C, A, B, bias, m, n, k, alpha, beta, transA, transB);
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  for (int i = 0; i < m * n; ++i)
    C[i] *= 0.5f * (1.f + std::tanh(kSqrt2OverPi * (C[i] + 0.044715f * C[i] * C[i] * C[i])));
}

// the (rows x cols) column-major matrix `dense` stored with leading dimension ld
// >= rows; the padding rows are filled with a poison value
static std::vector<float> padColMajor(const float *dense, int rows, int cols,
                                      int ld) {
  std::vector<float> padded(static_cast<size_t>(ld) * cols, 99.f);
  for (int j = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i)
      padded[j * ld + i] = dense[j * rows + i];
  return padded;
}

// batched multiply with an epilogue: C_b = epilogue(A_b * B_b + bias_b), X_b = X + b * strideX
// (column-major, A_b stored with leading dimension lda), bias_b a vector of m
static void refBatchedEpilogue(float *C, const float *A, int lda,
                               long long strideA, const float *B,
                               long long strideB, const float *bias,
                               long long strideBias, int m, int n, int k,
                               int batch, Epilogue epilogue) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  for (int b = 0; b < batch; ++b)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        float sum = 0.f;
        for (int p = 0; p < k; ++p)
          sum += A[b * strideA + p * lda + i] * B[b * strideB + j * k + p];
        sum += bias[b * strideBias + i];
        if (epilogue == Epilogue::ReluBias)
          sum = sum > 0.f ? sum : 0.f;
        else if (epilogue == Epilogue::GeluBias)
          sum *= 0.5f * (1.f + std::tanh(kSqrt2OverPi * (sum + 0.044715f * sum * sum * sum)));
        C[b * (long long)m * n + j * m + i] = sum;
      }
}

static int gFailures = 0;

static void checkClose(const float *got, const float *expected, int n,
                       const std::string &name, float rtol = 1e-4f,
                       float atol = 1e-4f) {
  bool pass = true;
  for (int i = 0; i < n; ++i) {
    float diff = std::abs(got[i] - expected[i]);
    float thr = atol + rtol * std::abs(expected[i]);
    if (diff > thr) {
      std::cerr << "  FAIL [" << name << "] idx=" << i << " got=" << got[i]
                << " expected=" << expected[i] << " diff=" << diff << "\n";
      pass = false;
    }
  }
  if (pass)
    std::cout << "  PASS  " << name << "\n";
  else
    ++gFailures;
}

static void fillSeq(float *M, int n, float start = 1.f, float step = 1.f) {
  for (int i = 0; i < n; ++i)
    M[i] = start + static_cast<float>(i) * step;
}

static void fillVal(float *M, int n, float v) {
  for (int i = 0; i < n; ++i)
    M[i] = v;
}

// C = op(A) * op(B), int8 inputs accumulated into int32 output (column-major)
static void refMatmulI8(int32_t *C, const int8_t *A, const int8_t *B, int m,
                        int n, int k, bool transA, bool transB) {
  int lda = transA ? k : m;
  int ldb = transB ? n : k;
  auto at = [&](const int8_t *M, int row, int col, int ld) {
    return static_cast<int32_t>(M[col * ld + row]);
  };
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      int32_t sum = 0;
      for (int p = 0; p < k; ++p) {
        int32_t a = transA ? at(A, p, i, lda) : at(A, i, p, lda);
        int32_t b = transB ? at(B, j, p, ldb) : at(B, p, j, ldb);
        sum += a * b;
      }
      C[j * m + i] = sum;
    }
  }
}

static void checkEqual(const int32_t *got, const int32_t *expected, int n,
                       const std::string &name) {
  bool pass = true;
  for (int i = 0; i < n; ++i) {
    if (got[i] != expected[i]) {
      std::cerr << "  FAIL [" << name << "] idx=" << i << " got=" << got[i]
                << " expected=" << expected[i] << "\n";
      pass = false;
    }
  }
  if (pass)
    std::cout << "  PASS  " << name << "\n";
  else
    ++gFailures;
}

static void fillSeqI8(int8_t *M, int n, int8_t start = 1, int8_t step = 1) {
  for (int i = 0; i < n; ++i)
    M[i] = static_cast<int8_t>(start + i * step);
}

// C = epilogue(alpha * op(A) * op(B) + beta * C + bias), with bias one value
// per row of C (unused by Default and Relu), column-major, in float
static void refEpilogue(float *C, const float *A, const float *B,
                        const float *bias, int m, int n, int k, float alpha,
                        float beta, bool transA, bool transB,
                        Epilogue epilogue) {
  refMatmul(C, A, B, m, n, k, alpha, beta, transA, transB);
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      float &x = C[j * m + i];
      if (epilogue != Epilogue::Default && epilogue != Epilogue::Relu)
        x += bias[i];
      if (epilogue == Epilogue::ReluBias || epilogue == Epilogue::Relu)
        x = x > 0.f ? x : 0.f;
      else if (epilogue == Epilogue::GeluBias)
        x *= 0.5f * (1.f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
    }
}

// int8 output: alpha * op(A) * op(B) + bias in float, clamped at 0 by the
// relu epilogues, rounded to the nearest integer and saturated to [-128, 127]
// (ties to even); alphaVec, when given, holds one alpha per row of C
static void refMatmulI8Out(int8_t *C, const int8_t *A, const int8_t *B,
                           const float *bias, int m, int n, int k, float alpha,
                           bool transA, bool transB, Epilogue epilogue,
                           const float *alphaVec = nullptr) {
  std::vector<int32_t> acc(static_cast<std::size_t>(m) * n);
  refMatmulI8(acc.data(), A, B, m, n, k, transA, transB);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      float x = (alphaVec ? alphaVec[i] : alpha) *
                static_cast<float>(acc[j * m + i]);
      if (epilogue == Epilogue::Bias || epilogue == Epilogue::ReluBias)
        x += bias[i];
      if (epilogue == Epilogue::ReluBias || epilogue == Epilogue::Relu)
        x = x > 0.f ? x : 0.f;
      x = std::nearbyint(x);
      C[j * m + i] = static_cast<int8_t>(std::min(127.f, std::max(-128.f, x)));
    }
}

// test data for the int8-output GEMMs: A (m x k after op) and B (k x n after
// op) stored dense, whatever the transposes
static void fillI8Pattern(std::vector<int8_t> &A, std::vector<int8_t> &B) {
  for (std::size_t i = 0; i < A.size(); ++i)
    A[i] = static_cast<int8_t>(static_cast<int>(i * 5 % 17) - 8);
  for (std::size_t i = 0; i < B.size(); ++i)
    B[i] = static_cast<int8_t>(static_cast<int>(i * 3 % 13) - 6);
}

static void checkEqualI8(const int8_t *got, const int8_t *expected, int n,
                         const std::string &name) {
  const std::vector<int32_t> g(got, got + n), e(expected, expected + n);
  checkEqual(g.data(), e.data(), n, name);
}

// OCP FP8 E4M3 (e4m3fn) bits to float, exact
static float refFp8ToFloat(uint8_t b) {
  const int exp = (b >> 3) & 0xF, mant = b & 0x7;
  float x;
  if (exp == 0xF && mant == 0x7)
    x = std::nanf("");
  else if (exp == 0)
    x = std::ldexp(static_cast<float>(mant), -9);
  else
    x = std::ldexp(1.f + static_cast<float>(mant) / 8.f, exp - 7);
  return (b & 0x80) ? -x : x;
}

// fp8 test values: magnitudes between 2^-3 and 2^3 with both signs
static std::vector<uint8_t> fp8Pattern(std::size_t n, unsigned seed) {
  std::vector<uint8_t> v(n);
  unsigned s = seed;
  for (auto &b : v) {
    s = s * 1103515245u + 12345u;
    const unsigned r = s >> 16;
    const unsigned exp = 4 + r % 7, mant = (r >> 3) & 0x7, sign = (r >> 6) & 1;
    b = static_cast<uint8_t>((sign << 7) | (exp << 3) | mant);
  }
  return v;
}

// the bf16 bits of x, which must be exact in bf16
static uint16_t bf16Bits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return static_cast<uint16_t>(u >> 16);
}

// reference of the fp8 GEMMs with a float C: the inputs decoded exactly, then
// C = epilogue(alpha * scaleA * scaleB * op(A) * op(B) + bias) in float
static std::vector<float> refFp8Gemm(const uint8_t *A, const uint8_t *B,
                                     const float *bias, int m, int n, int k,
                                     float alpha, float scaleA, float scaleB,
                                     bool transA, bool transB,
                                     Epilogue epilogue) {
  std::vector<float> fa(static_cast<std::size_t>(m) * k),
      fb(static_cast<std::size_t>(k) * n), c(static_cast<std::size_t>(m) * n);
  for (std::size_t i = 0; i < fa.size(); ++i)
    fa[i] = refFp8ToFloat(A[i]);
  for (std::size_t i = 0; i < fb.size(); ++i)
    fb[i] = refFp8ToFloat(B[i]);
  refEpilogue(c.data(), fa.data(), fb.data(), bias, m, n, k,
              alpha * scaleA * scaleB, 0.f, transA, transB, epilogue);
  return c;
}

// ---------------------------------------------------------------------------
// CPU tests
// ---------------------------------------------------------------------------

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#include "cpu/unit_test.tpp"

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
static int ldaFor(char trans, int m, int k) {
  return (trans == 'N' || trans == 'n') ? m : k;
}
static int ldbFor(char trans, int k, int n) {
  return (trans == 'N' || trans == 'n') ? k : n;
}

// host-side conversions with the fp16 helpers of cuda_fp16.h / hip_fp16.h
static std::vector<__half> toHalf(const std::vector<float> &v) {
  std::vector<__half> h(v.size());
  for (std::size_t i = 0; i < v.size(); ++i)
    h[i] = __float2half(v[i]);
  return h;
}

static std::vector<float> fromHalf(const __half *h, std::size_t n) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = __half2float(h[i]);
  return v;
}

#include "gpu/unit_test.tpp"
#endif

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  runCpuTests();
  runCpuQuantizedTests();
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  runGpuTests<alpaka::TagGpuCudaRt>();
  runGpuDynamicShapeTests<alpaka::TagGpuCudaRt>();
  runGpuLowPrecisionTests<alpaka::TagGpuCudaRt>();
  runGpuQuantizedTests<alpaka::TagGpuCudaRt>();
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  runGpuTests<alpaka::TagGpuHipRt>();
  runGpuDynamicShapeTests<alpaka::TagGpuHipRt>();
  runGpuLowPrecisionTests<alpaka::TagGpuHipRt>();
  runGpuQuantizedTests<alpaka::TagGpuHipRt>();
#endif

  std::cout << "\n";
  if (gFailures == 0)
    std::cout << "All tests passed.\n";
  else
    std::cout << gFailures << " test(s) FAILED.\n";

  return gFailures > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
