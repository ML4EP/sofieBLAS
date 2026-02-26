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
  if (err != cudaSuccess) {                                                    \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line "      \
              << __LINE__ << "\n";                                             \
    exit(EXIT_FAILURE);                                                        \
  }

#define CHECK_CUBLAS(status)                                                   \
  do {                                                                         \
    cublasStatus_t s = (status);                                               \
    if (s != CUBLAS_STATUS_SUCCESS) {                                          \
      std::cerr << "cuBLAS error " << s << " at line " << __LINE__             \
                << std::endl;                                                  \
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
  cublasHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  void *d_workspace = nullptr;
  size_t workspaceSize = 1 << 22; // 4MB
  cudaStream_t stream = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  int error_flag = 0;

  std::unordered_map<std::pair<std::size_t, std::size_t>,
                     cublasLtMatrixLayout_t, PairHash, PairEq>
      LayoutStore;

public:
    BlasCuda(const BlasCuda&) = delete;
    BlasCuda& operator=(const BlasCuda&) = delete;
    BlasCuda(BlasCuda&&) = delete;
    BlasCuda& operator=(BlasCuda&&) = delete;

  BlasCuda(alpaka::QueueCudaRtNonBlocking &queue) : m_queue{queue} {
    stream = static_cast<cudaStream_t>(m_queue.getNativeHandle());
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));
    CHECK_CUBLAS(cublasCreate(&handle));
    heuristic = {};
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F,
                                          CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }

  ~BlasCuda() {
    for (auto& [key, layout] : LayoutStore) {
      if (layout) {
        cublasLtMatrixLayoutDestroy(layout);
      }
    }
    LayoutStore.clear();

    if (preference)
      cublasLtMatmulPreferenceDestroy(preference);
    if (operationDesc)
      cublasLtMatmulDescDestroy(operationDesc);
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

  void AddLayoutConfig(std::size_t m, std::size_t n, std::size_t k) {
    CheckAndAddLayout(k, m); // A is m x k
    CheckAndAddLayout(k, n); // B is k x n
    CheckAndAddLayout(m, n); // C is m x n
    CheckAndAddLayout(1, m);
  }

template <typename T, typename TIdx>
inline void
gemm(char transa, char transb, const unsigned int m,
     const unsigned int n, const unsigned int k,
     const float alpha,
     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
     const float beta,
     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C)
{
    // Create a LOCAL descriptor — don't touch the shared member
    cublasLtMatmulDesc_t localDesc = nullptr;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&localDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transB_op = charToCuBlasTranspose(transb);
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        localDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB_op, sizeof(transB_op)));

    cublasOperation_t transA_op = charToCuBlasTranspose(transa);
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        localDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA_op, sizeof(transA_op)));

    void *bias_ptr = reinterpret_cast<void *>(alpaka::getPtrNative(bias));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        localDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));

    cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        localDesc,
        CUBLASLT_MATMUL_DESC_EPILOGUE,
        &ep,
        sizeof(ep)));


    cublasLtMatmulHeuristicResult_t localHeuristic{};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        localDesc,
        LayoutStore[{k, m}],
        LayoutStore[{k, n}],
        LayoutStore[{1, m}],
        LayoutStore[{m, n}],
        preference,
        1,
        &localHeuristic,
        &returnedResults));

    if (returnedResults == 0) {
        cublasLtMatmulDescDestroy(localDesc);
        std::cerr << "No suitable cuBLASLt algorithm found!\n";
        exit(EXIT_FAILURE);
    }

    CHECK_CUBLAS(cublasLtMatmul(
        ltHandle,
        localDesc,
        &alpha,
        alpaka::getPtrNative(A), LayoutStore[{k, m}],
        alpaka::getPtrNative(B), LayoutStore[{k, n}],
        &beta,
        alpaka::getPtrNative(bias), LayoutStore[{1, m}],
        alpaka::getPtrNative(C),    LayoutStore[{m, n}],
        &(localHeuristic.algo),
        d_workspace,
        workspaceSize,
        stream));

      cudaDeviceSynchronize();

    // Safe to destroy AFTER submitting to stream — the handle is host-side,
    // the algo selection is already baked into the submitted work
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(localDesc));
}

template <typename T, typename TIdx>
inline void
gemmrelu(char transa, char transb, const unsigned int m,
     const unsigned int n, const unsigned int k,
     const float alpha,
     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
     const float beta,
     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C)
{
    // Create a LOCAL descriptor — don't touch the shared member
    cublasLtMatmulDesc_t localDesc = nullptr;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&localDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transB_op = charToCuBlasTranspose(transb);
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        localDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB_op, sizeof(transB_op)));

    cublasOperation_t transA_op = charToCuBlasTranspose(transa);
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        localDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA_op, sizeof(transA_op)));

    void *bias_ptr = reinterpret_cast<void *>(alpaka::getPtrNative(bias));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        localDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));

    cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_RELU_BIAS;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        localDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)));

    cublasLtMatmulHeuristicResult_t localHeuristic{};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        localDesc,
        LayoutStore[{k, m}],
        LayoutStore[{k, n}],
        LayoutStore[{1, m}],
        LayoutStore[{m, n}],
        preference,
        1,
        &localHeuristic,
        &returnedResults));

    if (returnedResults == 0) {
        cublasLtMatmulDescDestroy(localDesc);
        std::cerr << "No suitable cuBLASLt algorithm found!\n";
        exit(EXIT_FAILURE);
    }

    CHECK_CUBLAS(cublasLtMatmul(
        ltHandle,
        localDesc,
        &alpha,
        alpaka::getPtrNative(A), LayoutStore[{k, m}],
        alpaka::getPtrNative(B), LayoutStore[{k, n}],
        &beta,
        alpaka::getPtrNative(bias), LayoutStore[{1, m}],
        alpaka::getPtrNative(C),    LayoutStore[{m, n}],
        &(localHeuristic.algo),
        d_workspace,
        workspaceSize,
        stream));

              cudaDeviceSynchronize();

    // Safe to destroy AFTER submitting to stream — the handle is host-side,
    // the algo selection is already baked into the submitted work
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(localDesc));
}

  template <typename T, typename TIdx>
  inline void gemmgelu(char transa, char transb, const unsigned int m,
                       const unsigned int n, const unsigned int k,
                       const float alpha,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                       const float beta,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {

    void *bias_ptr = reinterpret_cast<void *>(bias.data());
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
        sizeof(bias_ptr)));

    cublasOperation_t transB = charToCuBlasTranspose(transb);
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    cublasOperation_t transA = charToCuBlasTranspose(transa);
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));

    SetGeluActivation();

    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, 
        LayoutStore[{k, m}],            // Adesc (m x k)
        LayoutStore[{k, n}],            // Bdesc (n x k)
        LayoutStore[{m, n}],            // Ddesc (m x 1)
        LayoutStore[{m, n}],            // Ddesc (m x n)
        preference, 1, &heuristic, &error_flag));
    if (error_flag == 0) {
      std::cerr << "No suitable cuBLASLt algorithm found!\n";
      exit(EXIT_FAILURE);
    }

    CHECK_CUBLAS(cublasLtMatmul(
        ltHandle, operationDesc, &alpha, A.data(), LayoutStore[{k, m}],
        B.data(), LayoutStore[{k, n}], &beta, bias.data(), LayoutStore[{m, n}],
        C.data(), LayoutStore[{m, n}], &(heuristic.algo), d_workspace,
        workspaceSize, stream));
  }

private:
  alpaka::QueueCudaRtNonBlocking m_queue;

  void CheckAndAddLayout(size_t rows, size_t cols) {
    auto key = std::make_pair(rows, cols);
    if (LayoutStore.find(key) == LayoutStore.end()) {
      cublasLtMatrixLayout_t temp = nullptr;
      size_t ld = rows;
      CHECK_CUBLAS(
          cublasLtMatrixLayoutCreate(&temp, CUDA_R_32F, rows, cols, ld));
      LayoutStore.emplace(key, temp);
    }
  }

  void ResetActivation() {
    epilogue = CUBLASLT_EPILOGUE_BIAS;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
  }

  void SetReluActivation() {
    epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
  }

  void SetGeluActivation() {
    epilogue = CUBLASLT_EPILOGUE_GELU;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
  }
};

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuCudaRt> {
public:
  using Impl = BlasCuda;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
