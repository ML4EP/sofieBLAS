#include "sofieBLAS/sofieBLAS.hpp"
#include <alpaka/alpaka.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
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
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  for (int i = 0; i < m * n; ++i)
    C[i] *= 0.5f * (1.f + std::erff(C[i] * kInvSqrt2));
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
  constexpr float kInvSqrt2 = 0.7071067811865476f;
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
          sum *= 0.5f * (1.f + std::erff(sum * kInvSqrt2));
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
#include "gpu/unit_test.tpp"
#endif

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  runCpuTests();
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  runGpuTests<alpaka::TagGpuCudaRt>();
  runGpuDynamicShapeTests<alpaka::TagGpuCudaRt>();
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  runGpuTests<alpaka::TagGpuHipRt>();
  runGpuDynamicShapeTests<alpaka::TagGpuHipRt>();
#endif

  std::cout << "\n";
  if (gFailures == 0)
    std::cout << "All tests passed.\n";
  else
    std::cout << gFailures << " test(s) FAILED.\n";

  return gFailures > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
