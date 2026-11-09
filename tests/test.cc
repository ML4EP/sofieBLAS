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
  constexpr float kSqrt2_OverPi = 0.7978845608028654f;
  for (int i = 0; i < m * n; ++i)
    C[i] *= 0.5f * (1.f + std::tanh(kSqrt2_OverPi *
                                    (C[i] + 0.044715f * C[i] * C[i] * C[i])));
}

// the (rows x cols) column-major matrix `dense` stored with leading dimension
// ld
// >= rows; the padding rows are filled with a poison value
static std::vector<float> padColMajor(const float *dense, int rows, int cols,
                                      int ld) {
  std::vector<float> padded(static_cast<size_t>(ld) * cols, 99.f);
  for (int j = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i)
      padded[j * ld + i] = dense[j * rows + i];
  return padded;
}

// batched multiply with an epilogue: C_b = epilogue(A_b * B_b + bias_b), X_b =
// X + b * strideX (column-major, A_b stored with leading dimension lda), bias_b
// a vector of m
static void refBatchedEpilogue(float *C, const float *A, int lda,
                               long long strideA, const float *B,
                               long long strideB, const float *bias,
                               long long strideBias, int m, int n, int k,
                               int batch, Epilogue epilogue) {
  constexpr float kSqrt2_OverPi = 0.7978845608028654f;
  for (int b = 0; b < batch; ++b)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        float sum = 0.f;
        for (int p = 0; p < k; ++p)
          sum += A[b * strideA + p * lda + i] * B[b * strideB + j * k + p];
        if (epilogue != Epilogue::Default && epilogue != Epilogue::Relu &&
            epilogue != Epilogue::Gelu)
          sum += bias[b * strideBias + i];
        if (epilogue == Epilogue::ReluBias || epilogue == Epilogue::Relu)
          sum = sum > 0.f ? sum : 0.f;
        else if (epilogue == Epilogue::GeluBias || epilogue == Epilogue::Gelu)
          sum *= 0.5f * (1.f + std::tanh(kSqrt2_OverPi *
                                         (sum + 0.044715f * sum * sum * sum)));
        C[b * (long long)m * n + j * m + i] = sum;
      }
}

static int gFailures = 0;

template <typename TBlas, typename TQueue>
static void checkQuantizationOption(TQueue &queue, const std::string &name) {
  TBlas plain(queue);
  bool threw = false;
  try {
    plain.int8Matmul('N', 'N', 1u, 1u, 1u, static_cast<const int8_t *>(nullptr),
                     static_cast<const int8_t *>(nullptr),
                     static_cast<int32_t *>(nullptr));
  } catch (const std::logic_error &) {
    threw = true;
  }
  TBlas quantized(queue, {.quantization = true});
  if (threw && !plain.quantizationEnabled() &&
      quantized.quantizationEnabled()) {
    std::cout << "  PASS  " << name << "\n";
  } else {
    std::cerr << "  FAIL [" << name << "]\n";
    ++gFailures;
  }
}

static void checkClose(const float *actual, const float *expected, int n,
                       const std::string &name, float rtol = 1e-4f,
                       float atol = 1e-4f) {
  bool pass = true;
  for (int i = 0; i < n; ++i) {
    float diff = std::abs(actual[i] - expected[i]);
    float thr = atol + rtol * std::abs(expected[i]);
    if (diff > thr) {
      std::cerr << "  FAIL [" << name << "] idx=" << i
                << " actual=" << actual[i] << " expected=" << expected[i]
                << " diff=" << diff << "\n";
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

static void checkEqual(const int32_t *actual, const int32_t *expected, int n,
                       const std::string &name, int tol = 0) {
  bool pass = true;
  for (int i = 0; i < n; ++i) {
    if (std::abs(actual[i] - expected[i]) > tol) {
      std::cerr << "  FAIL [" << name << "] idx=" << i
                << " actual=" << actual[i] << " expected=" << expected[i]
                << "\n";
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
  constexpr float kSqrt2_OverPi = 0.7978845608028654f;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      float &x = C[j * m + i];
      if (epilogue != Epilogue::Default && epilogue != Epilogue::Relu &&
          epilogue != Epilogue::Gelu)
        x += bias[i];
      if (epilogue == Epilogue::ReluBias || epilogue == Epilogue::Relu)
        x = x > 0.f ? x : 0.f;
      else if (epilogue == Epilogue::GeluBias || epilogue == Epilogue::Gelu)
        x *= 0.5f *
             (1.f + std::tanh(kSqrt2_OverPi * (x + 0.044715f * x * x * x)));
    }
}

// int8 output: alpha * op(A) * op(B) + bias in float, clamped at 0 by the
// relu epilogues, rounded to the nearest integer and saturated to [-128, 127]
// (ties to even); alphaVec, when given, holds one alpha per row of C
static void refMatmulI8Out(int8_t *C, const int8_t *A, const int8_t *B,
                           const float *bias, int m, int n, int k, float alpha,
                           bool transA, bool transB, Epilogue epilogue,
                           const float *alphaVec = nullptr,
                           float *unrounded = nullptr) {
  std::vector<int32_t> acc(static_cast<std::size_t>(m) * n);
  refMatmulI8(acc.data(), A, B, m, n, k, transA, transB);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      float x =
          (alphaVec ? alphaVec[i] : alpha) * static_cast<float>(acc[j * m + i]);
      if (epilogue == Epilogue::Bias || epilogue == Epilogue::ReluBias ||
          epilogue == Epilogue::GeluBias)
        x += bias[i];
      if (epilogue == Epilogue::ReluBias || epilogue == Epilogue::Relu)
        x = x > 0.f ? x : 0.f;
      else if (epilogue == Epilogue::GeluBias || epilogue == Epilogue::Gelu)
        x *= 0.5f * (1.f + std::tanh(0.7978845608028654f *
                                     (x + 0.044715f * x * x * x)));
      if (unrounded)
        unrounded[j * m + i] = x;
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

// exact match, except that an element may differ by 1 where the unrounded
// reference lies within eps of, but not exactly on, a rounding tie (x.5): there
// a last-ulp difference in the float computation (GELU, fused multiply-add, a
// non-exact alpha) flips the rounding. An exact tie must match exactly, so a
// wrong rounding mode is still caught. Used for every int8 output.
static void checkEqualI8Ties(const int8_t *actual, const int8_t *expected,
                             const float *unrounded, int n,
                             const std::string &name, float eps = 1e-3f) {
  bool pass = true;
  int flips = 0;
  for (int i = 0; i < n; ++i) {
    const int d = std::abs(int(actual[i]) - int(expected[i]));
    if (d == 0)
      continue;
    const float frac = unrounded[i] - std::floor(unrounded[i]);
    const float distToTie = std::abs(frac - 0.5f);
    // an exact tie must round like the reference (half to even)
    if (d == 1 && distToTie > 0.f && distToTie <= eps) {
      ++flips;
      continue;
    }
    std::cerr << "  FAIL [" << name << "] idx=" << i
              << " actual=" << int(actual[i])
              << " expected=" << int(expected[i])
              << " unrounded=" << unrounded[i] << "\n";
    pass = false;
  }
  if (pass)
    std::cout << "  PASS  " << name << " (" << flips << " tie flips)\n";
  else
    ++gFailures;
}

static float refFp8ToFloat(uint8_t b, bool fnuz = false) {
  const int exp = (b >> 3) & 0xF, mant = b & 0x7;
  float x;
  if (fnuz) {
    if (b == 0x80)
      return std::nanf("");
    x = exp == 0 ? std::ldexp(static_cast<float>(mant), -10)
                 : std::ldexp(1.f + static_cast<float>(mant) / 8.f, exp - 8);
  } else if (exp == 0xF && mant == 0x7) {
    x = std::nanf("");
  } else if (exp == 0) {
    x = std::ldexp(static_cast<float>(mant), -9);
  } else {
    x = std::ldexp(1.f + static_cast<float>(mant) / 8.f, exp - 7);
  }
  return (b & 0x80) ? -x : x;
}

// the largest finite E4M3 value
static float refFp8Max(bool fnuz = false) { return fnuz ? 240.f : 448.f; }

// fp8 test values: normal numbers (exponent field 4 to 10) with both signs,
// valid in the OCP and the FNUZ format
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
                                     Epilogue epilogue, bool fnuz = false) {
  std::vector<float> fa(static_cast<std::size_t>(m) * k),
      fb(static_cast<std::size_t>(k) * n), c(static_cast<std::size_t>(m) * n);
  for (std::size_t i = 0; i < fa.size(); ++i)
    fa[i] = refFp8ToFloat(A[i], fnuz);
  for (std::size_t i = 0; i < fb.size(); ++i)
    fb[i] = refFp8ToFloat(B[i], fnuz);
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
  runCpuFp16Tests();
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  runGpuTests<alpaka::TagGpuCudaRt>();
  runGpuDynamicShapeTests<alpaka::TagGpuCudaRt>();
  runGpuLowPrecisionTests<alpaka::TagGpuCudaRt>();
  runGpuQuantizedTests<alpaka::TagGpuCudaRt>();
  runGpuTests<alpaka::TagGpuCudaRt>(
      {.quantization = true, .forceNative = true});
  runGpuLowPrecisionTests<alpaka::TagGpuCudaRt>(
      {.quantization = true, .forceNative = true});
  runGpuQuantizedTests<alpaka::TagGpuCudaRt>(
      {.quantization = true, .forceNative = true});
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  runGpuTests<alpaka::TagGpuHipRt>();
  runGpuDynamicShapeTests<alpaka::TagGpuHipRt>();
  runGpuLowPrecisionTests<alpaka::TagGpuHipRt>();
  runGpuQuantizedTests<alpaka::TagGpuHipRt>();
  runGpuTests<alpaka::TagGpuHipRt>({.quantization = true, .forceNative = true});
  runGpuLowPrecisionTests<alpaka::TagGpuHipRt>(
      {.quantization = true, .forceNative = true});
  runGpuQuantizedTests<alpaka::TagGpuHipRt>(
      {.quantization = true, .forceNative = true});
#endif

  std::cout << "\n";
  if (gFailures == 0)
    std::cout << "All tests passed.\n";
  else
    std::cout << gFailures << " test(s) FAILED.\n";

  return gFailures > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
