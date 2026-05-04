#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>
#include <cublasLt.h>
#include <cublas_v2.h>

#define CHECK_CUDA(err)                                                        \
  if ((err) != cudaSuccess) {                                                  \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line "      \
              << __LINE__ << "\n";                                             \
    exit(EXIT_FAILURE);                                                        \
  }

#define CHECK_CUBLAS(status)                                                   \
  do {                                                                         \
    cublasStatus_t _s = (status);                                              \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
      std::cerr << "cuBLAS error " << _s << " at line " << __LINE__ << "\n";   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

struct PairHash {
  std::size_t
  operator()(const std::pair<std::size_t, std::size_t> &p) const noexcept {
    std::size_t h1 = std::hash<std::size_t>{}(p.first);
    std::size_t h2 = std::hash<std::size_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct PairEq {
  bool operator()(const std::pair<std::size_t, std::size_t> &a,
                  const std::pair<std::size_t, std::size_t> &b) const noexcept {
    return a.first == b.first && a.second == b.second;
  }
};

class BlasCuda {
  cublasLtHandle_t ltHandle = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  void *d_workspace = nullptr;
  size_t workspaceSize = 1 << 22; // 4 MB
  cudaStream_t stream = nullptr;
  std::unordered_map<std::pair<std::size_t, std::size_t>,
                     cublasLtMatrixLayout_t, PairHash, PairEq>
      layoutStore;

public:
  BlasCuda(const BlasCuda &) = delete;
  BlasCuda &operator=(const BlasCuda &) = delete;
  BlasCuda(BlasCuda &&) = delete;
  BlasCuda &operator=(BlasCuda &&) = delete;

  BlasCuda(alpaka::QueueCudaRtNonBlocking &queue) : m_queue{queue} {
    stream = static_cast<cudaStream_t>(m_queue.getNativeHandle());
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }

  ~BlasCuda() {
    for (auto &[key, layout] : layoutStore)
      if (layout)
        cublasLtMatrixLayoutDestroy(layout);
    if (preference)
      cublasLtMatmulPreferenceDestroy(preference);
    if (ltHandle)
      cublasLtDestroy(ltHandle);
    if (d_workspace)
      cudaFree(d_workspace);
  }

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

  // Register matrix layouts for a given (m, n, k, lda, ldb, ldc, transa,
  // transb). Must be called before gemm/gemmrelu/gemmgelu/matmul for each
  // unique combination.
  void addLayoutConfig(std::size_t m, std::size_t n, std::size_t k,
                       std::size_t lda, std::size_t ldb, std::size_t ldc,
                       char transa, char transb) {
    // Physical A: (m×k) if NoTrans, (k×m) if Trans
    if (transa == 'N' || transa == 'n')
      checkAndAddLayout(m, k, lda);
    else
      checkAndAddLayout(k, m, lda);
    // Physical B: (k×n) if NoTrans, (n×k) if Trans
    if (transb == 'N' || transb == 'n')
      checkAndAddLayout(k, n, ldb);
    else
      checkAndAddLayout(n, k, ldb);
    // C is always (m×n)
    checkAndAddLayout(m, n, ldc);
  }

  // C = alpha * op(A) * op(B) + beta * bias + bias_vec  (bias_vec broadcast per
  // row)
  template <typename T, typename TIdx>
  inline void
  gemm(char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
       float alpha, alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B, float beta,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    const void *bptr = alpaka::getPtrNative(bias);
    auto desc =
        makeDesc(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                 CUBLASLT_EPILOGUE_BIAS, bptr);
    executeMatmul(desc, alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B),
                  beta, alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  // C = relu(alpha * op(A) * op(B) + beta * bias + bias_vec)
  template <typename T, typename TIdx>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    const void *bptr = alpaka::getPtrNative(bias);
    auto desc =
        makeDesc(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                 CUBLASLT_EPILOGUE_RELU_BIAS, bptr);
    executeMatmul(desc, alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B),
                  beta, alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  // C = gelu(alpha * op(A) * op(B) + beta * bias + bias_vec)
  template <typename T, typename TIdx>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    const void *bptr = alpaka::getPtrNative(bias);
    auto desc =
        makeDesc(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                 CUBLASLT_EPILOGUE_GELU_BIAS, bptr);
    executeMatmul(desc, alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B),
                  beta, alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  // C = alpha * op(A) * op(B) + beta * C  (no bias)
  template <typename T, typename TIdx>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha,
                     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                     float beta,
                     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    auto desc =
        makeDesc(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                 CUBLASLT_EPILOGUE_DEFAULT);
    float *c = alpaka::getPtrNative(C);
    executeMatmul(desc, alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B),
                  beta, c, c, layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  // matmul on raw pointers directly (no layout caching, caller must ensure
  // correct leading dims and layout)
  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const * A, T const * B,
                     float beta, T * C) {
    auto desc =
        makeDesc(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                 CUBLASLT_EPILOGUE_DEFAULT);
    executeMatmul(desc, alpha, A, B, beta, C, C, layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

private:
  alpaka::QueueCudaRtNonBlocking m_queue;

  // Returns the layout map key for matrix A based on its transpose flag.
  // Physical dimensions: NoTrans → (m×k), Trans → (k×m).
  static std::pair<std::size_t, std::size_t>
  layoutKeyA(char trans, std::size_t m, std::size_t k) {
    return (trans == 'N' || trans == 'n') ? std::make_pair(m, k)
                                          : std::make_pair(k, m);
  }

  // Returns the layout map key for matrix B based on its transpose flag.
  // Physical dimensions: NoTrans → (k×n), Trans → (n×k).
  static std::pair<std::size_t, std::size_t>
  layoutKeyB(char trans, std::size_t k, std::size_t n) {
    return (trans == 'N' || trans == 'n') ? std::make_pair(k, n)
                                          : std::make_pair(n, k);
  }

  void checkAndAddLayout(std::size_t rows, std::size_t cols, std::size_t ld) {
    auto key = std::make_pair(rows, cols);
    if (layoutStore.find(key) == layoutStore.end()) {
      cublasLtMatrixLayout_t layout = nullptr;
      CHECK_CUBLAS(
          cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, rows, cols, ld));
      layoutStore.emplace(key, layout);
    }
  }

  // Creates a per-call matmul descriptor with transpose ops, epilogue, and
  // optional bias pointer. Caller owns the returned descriptor.
  cublasLtMatmulDesc_t makeDesc(cublasOperation_t transA,
                                cublasOperation_t transB,
                                cublasLtEpilogue_t epilogue,
                                const void *bias_ptr = nullptr) {
    cublasLtMatmulDesc_t desc = nullptr;
    CHECK_CUBLAS(
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    if (bias_ptr) {
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
          sizeof(bias_ptr)));
    }
    return desc;
  }

  // Runs heuristic selection, executes cublasLtMatmul, syncs stream, and
  // destroys desc. D_in is the matrix scaled by beta (may equal C_out for
  // in-place accumulation).
  void executeMatmul(cublasLtMatmulDesc_t desc, float alpha, const float *A,
                     const float *B, float beta, const float *D_in,
                     float *C_out,
                     const std::pair<std::size_t, std::size_t> &kA,
                     const std::pair<std::size_t, std::size_t> &kB,
                     const std::pair<std::size_t, std::size_t> &kC) {
    cublasLtMatmulHeuristicResult_t h{};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, desc, layoutStore.at(kA), layoutStore.at(kB),
        layoutStore.at(kC), layoutStore.at(kC), preference, 1, &h,
        &returnedResults));
    if (returnedResults == 0) {
      cublasLtMatmulDescDestroy(desc);
      std::cerr << "No suitable cuBLASLt algorithm found!\n";
      exit(EXIT_FAILURE);
    }
    CHECK_CUBLAS(cublasLtMatmul(ltHandle, desc, &alpha, A, layoutStore.at(kA),
                                B, layoutStore.at(kB), &beta, D_in,
                                layoutStore.at(kC), C_out, layoutStore.at(kC),
                                &h.algo, d_workspace, workspaceSize, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(desc));
  }
};

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuCudaRt> {
public:
  using Impl = BlasCuda;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
