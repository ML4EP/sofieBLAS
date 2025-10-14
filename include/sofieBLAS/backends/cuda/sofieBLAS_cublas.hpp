#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <unordered_map>
#include <utility>
#include <functional>
#include <iostream>
#include <cstdlib>

#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>
#include <cublasLt.h>
#include <cublas_v2.h>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUBLAS(status) \
    do { \
        cublasStatus_t s = (status); \
        if (s != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error " << s << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// --- Pair hash / eq functors (for std::pair<size_t,size_t>) ---
struct PairHash {
    std::size_t operator()(const std::pair<std::size_t, std::size_t>& p) const noexcept {
        std::size_t h1 = std::hash<std::size_t>{}(p.first);
        std::size_t h2 = std::hash<std::size_t>{}(p.second);
        // boost-like combine
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

struct PairEq {
    bool operator()(const std::pair<std::size_t, std::size_t>& a,
                    const std::pair<std::size_t, std::size_t>& b) const noexcept {
        return a.first == b.first && a.second == b.second;
    }
};

// --- LayoutCombin (key) and separate hash / eq functors ---
struct LayoutCombin {
    std::pair<size_t, size_t> a;
    std::pair<size_t, size_t> b;
    std::pair<size_t, size_t> c;

    bool operator==(const LayoutCombin& other) const {
        return a == other.a && b == other.b && c == other.c;
    }
};

struct LayoutCombinHash {
    std::size_t operator()(const LayoutCombin& key) const noexcept {
        PairHash ph;
        std::size_t h1 = ph(key.a);
        std::size_t h2 = ph(key.b);
        std::size_t h3 = ph(key.c);
        // combine the three hashes
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct LayoutCombinEq {
    bool operator()(const LayoutCombin& x, const LayoutCombin& y) const noexcept {
        return x == y;
    }
};

struct LayoutAlgo {
    cublasLtMatrixLayout_t layout;
    cublasLtMatmulAlgo_t algo;
};

class BlasCuda {
    cublasLtHandle_t ltHandle = nullptr;
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    void* d_workspace = nullptr;
    size_t workspaceSize = 1 << 22; // 4MB
    cudaStream_t stream = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

    // use distinct hash and equal functors
    std::unordered_map<
        std::pair<std::size_t, std::size_t>,
        cublasLtMatrixLayout_t,
        PairHash,
        PairEq
    > LayoutStore;

    std::unordered_map<
        LayoutCombin,
        cublasLtMatmulAlgo_t,
        LayoutCombinHash,
        LayoutCombinEq
    > LayoutAlgoStore;

  public:
  BlasCuda(alpaka::QueueCudaRtNonBlocking &queue) : m_queue{queue} {
    stream = static_cast<cudaStream_t>(m_queue.getNativeHandle());
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));
    heuristic = {}; // zero-initialize
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize,
        sizeof(workspaceSize)));
  }

  ~BlasCuda() {
    // destroy in reverse order, free workspace
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
    if (ltHandle) cublasLtDestroy(ltHandle);
    if (d_workspace) cudaFree(d_workspace);
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

  void AddLayoutConfig(std::size_t m, std::size_t n, std::size_t k){
      CheckAndAddLayout(m, n);
      CheckAndAddLayout(n, k);
      CheckAndAddLayout(m, k);
      CheckAndAddLayoutAlgo({{m, n}, {n, k}, {m, k}});
  }

  template <typename T, typename TIdx>
  inline void
  gemmrelu(char transa, char transb, const unsigned int m, const unsigned int n,
       const unsigned int k, const float alpha,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
       const float beta,  alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {

    SetReluActivation();
  void* bias_ptr = reinterpret_cast<void*>(bias.data());
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &bias_ptr,
      sizeof(bias_ptr)));
    // IMPORTANT: LayoutStore[...] already yields cublasLtMatrixLayout_t (a pointer type).
    // Do NOT take & on it (that would create a pointer-to-pointer).
    CHECK_CUBLAS(cublasLtMatmul(
        ltHandle,
        operationDesc,
        &alpha,
        A.data(), LayoutStore[{m, n}],
        B.data(), LayoutStore[{n, k}],
        &beta,
        bias.data(), LayoutStore[{m, k}],
        C.data(), LayoutStore[{m, k}],
        &LayoutAlgoStore[{{m, n}, {n, k}, {m, k}}], // algorithm expects pointer to algo
        d_workspace,
        workspaceSize,
        stream));
  }

private:
  alpaka::QueueCudaRtNonBlocking m_queue;
  // removed unused m_handle

  void CheckAndAddLayout(size_t rows, size_t cols){
    auto key = std::make_pair(rows, cols);
    if (LayoutStore.find(key) == LayoutStore.end()) {
      cublasLtMatrixLayout_t temp = nullptr;
      // leading dimension = rows (row-major assuming your data layout). adjust if needed.
      CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&temp, CUDA_R_32F, rows, cols, rows));
      LayoutStore.emplace(key, temp);
    }
  }

  void CheckAndAddLayoutAlgo(LayoutCombin p){
    if (LayoutAlgoStore.find(p) == LayoutAlgoStore.end()) {
      int returnedResults = 0;
      // pass the stored layout *values* (not &LayoutStore[...] which would be pointer-to-pointer)
      CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
          ltHandle,
          operationDesc,
          LayoutStore[p.a], // Adesc
          LayoutStore[p.b], // Bdesc
          LayoutStore[p.c], // Cdesc
          LayoutStore[p.c], // Ddesc (same as C here)
          preference,
          1,
          &heuristic,
          &returnedResults));
      if (returnedResults == 0) { std::cerr << "No suitable cuBLASLt algorithm found!\n"; exit(EXIT_FAILURE); }
      LayoutAlgoStore[p] = heuristic.algo;
    }
  }

  void SetReluActivation(){
    // set epilogue to RELU+BIA S and propagate to matmulDesc
    epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  }

  void SetGeluActivation(){
    epilogue = CUBLASLT_EPILOGUE_GELU;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  }

};

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuCudaRt> {
public:
  using Impl = BlasCuda;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
