#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

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

struct DescKey {
  int transA; // CUBLAS_OP_N / CUBLAS_OP_T encoded as int
  int transB;
  int epilogue; // cublasLtEpilogue_t encoded as int
  bool operator==(const DescKey &o) const noexcept {
    return transA == o.transA && transB == o.transB && epilogue == o.epilogue;
  }
};

struct DescKeyHash {
  std::size_t operator()(const DescKey &k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.transA) * 97u +
                    static_cast<std::size_t>(k.transB) * 31u +
                    static_cast<std::size_t>(k.epilogue);
    return h ^ (h >> 16);
  }
};

struct AlgoKey {
  DescKey dk;
  std::size_t rowsA, colsA;
  std::size_t rowsB, colsB;
  bool operator==(const AlgoKey &o) const noexcept {
    return dk == o.dk && rowsA == o.rowsA && colsA == o.colsA &&
           rowsB == o.rowsB && colsB == o.colsB;
  }
};

struct AlgoKeyHash {
  std::size_t operator()(const AlgoKey &k) const noexcept {
    std::size_t h = DescKeyHash{}(k.dk);
    auto mix = [&](std::size_t v) {
      h ^= std::hash<std::size_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) +
           (h >> 2);
    };
    mix(k.rowsA);
    mix(k.colsA);
    mix(k.rowsB);
    mix(k.colsB);
    return h;
  }
};

// A call site's declared maximum shape, recorded by addLayoutConfig.
struct ShapeEnvelope {
  std::size_t rowsA, colsA, rowsB, colsB, rowsC, colsC;
};

struct LayoutStats {
  std::size_t heuristicQueries = 0; // algorithm searches issued
  std::size_t envelopeRejects = 0;  // declared-shape algo unusable at a call
  std::size_t evictions = 0;        // entries dropped to stay under the limit
};

class BlasCuda {
  cublasLtHandle_t ltHandle = nullptr;
  cublasHandle_t handle = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  void *d_workspace = nullptr;
  size_t workspaceSize = 1u << 25; // 32 MB
  cudaStream_t stream = nullptr;

  // One persistent layout descriptor per GEMM operand: ROLE_A and ROLE_B are
  // the input matrices of C = alpha * op(A) * op(B) + beta * C, ROLE_C the
  // output (cublasLtMatmul takes it twice, as C and D). stampLayout rewrites
  // each descriptor in place to the shape of the call at hand, which is what
  // lets one instance serve any runtime size.
  enum LayoutRole { ROLE_A = 0, ROLE_B = 1, ROLE_C = 2 };
  cublasLtMatrixLayout_t roleLayout[3] = {};

  std::unordered_map<DescKey, cublasLtMatmulDesc_t, DescKeyHash> descStore;

  // algo cache entry
  struct CacheEntry {
    cublasLtMatmulHeuristicResult_t h{};
    // position in lruOrder; only valid when a limit is set
    std::list<AlgoKey>::iterator lru{};
  };
  std::unordered_map<AlgoKey, CacheEntry, AlgoKeyHash> algoCache;
  std::list<AlgoKey> lruOrder;
  // 0 = unbounded
  std::size_t algoCacheLimit = 0;

  std::vector<ShapeEnvelope> envelopes;

  LayoutStats stats;

public:
  const LayoutStats &layoutStats() const { return stats; }
  std::size_t algoCacheSize() const { return algoCache.size(); }

  BlasCuda(const BlasCuda &) = delete;
  BlasCuda &operator=(const BlasCuda &) = delete;
  BlasCuda(BlasCuda &&) = delete;
  BlasCuda &operator=(BlasCuda &&) = delete;

  BlasCuda(alpaka::QueueCudaRtNonBlocking &queue,
           std::size_t algoCacheLimit_ = 0)
      : algoCacheLimit{algoCacheLimit_}, m_queue{queue} {
    stream = static_cast<cudaStream_t>(m_queue.getNativeHandle());

    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }

  ~BlasCuda() {
    for (auto L : roleLayout)
      if (L)
        cublasLtMatrixLayoutDestroy(L);
    for (auto &[key, desc] : descStore)
      if (desc)
        cublasLtMatmulDescDestroy(desc);
    if (preference)
      cublasLtMatmulPreferenceDestroy(preference);
    if (ltHandle)
      cublasLtDestroy(ltHandle);
    if (handle)
      cublasDestroy(handle);
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

  // Declares a call site's largest shape (its envelope) and resolves the
  // algorithm for it up front; the generated Session constructor calls this
  // once per GEMM call site with its construction-time dimensions. Which
  // epilogue the site will use is unknown here, so all three used by the
  // generated code are resolved; unused ones cost one heuristic query each,
  // off the inference path.
  void addLayoutConfig(std::size_t m, std::size_t n, std::size_t k, std::size_t,
                       std::size_t, std::size_t, char transa, char transb) {
    const auto shapeA = layoutKeyA(transa, m, k);
    const auto shapeB = layoutKeyB(transb, k, n);
    const std::pair<std::size_t, std::size_t> shapeC{m, n};
    envelopes.push_back(
        {shapeA.first, shapeA.second, shapeB.first, shapeB.second, m, n});

    const cublasOperation_t tA = charToCuBlasTranspose(transa);
    const cublasOperation_t tB = charToCuBlasTranspose(transb);
    const cublasLtEpilogue_t eps[] = {CUBLASLT_EPILOGUE_DEFAULT,
                                      CUBLASLT_EPILOGUE_BIAS,
                                      CUBLASLT_EPILOGUE_RELU_BIAS};
    for (cublasLtEpilogue_t ep : eps) {
      getOrComputeAlgo(tA, tB, ep, shapeA, shapeB, shapeC, /*required=*/false);
    }
  }

  template <typename T, typename TIdx>
  inline void
  gemm(char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
       float alpha, alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B, float beta,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_BIAS, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemm(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const
          &A,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const
          &B,
      float beta,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx>
          &bias,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_BIAS, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, T const *B,
                   float beta, T *bias, T *C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_BIAS, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_RELU_BIAS, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmrelu(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const
          &A,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const
          &B,
      float beta,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx>
          &bias,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_RELU_BIAS, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_RELU_BIAS, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_GELU_BIAS, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmgelu(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const
          &A,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const
          &B,
      float beta,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx>
          &bias,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_GELU_BIAS, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_GELU_BIAS, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha,
                     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                     float beta,
                     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    float *c = alpaka::getPtrNative(C);
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_DEFAULT, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, c, c, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void matmul(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const
          &A,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const
          &B,
      float beta,
      alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    T *c = alpaka::getPtrNative(C);
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_DEFAULT, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, c, c, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  // Raw-pointer overload
  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, T const *B,
                     float beta, T *C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_DEFAULT, alpha, A, B, beta, C, C, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  inline void gemmStridedBatched(char transa, char transb, int m, int n, int k,
                                 float alpha, const float *A, int lda,
                                 long long strideA, const float *B, int ldb,
                                 long long strideB, float beta, float *C,
                                 int ldc, long long strideC, int batchCount) {
    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle, charToCuBlasTranspose(transa), charToCuBlasTranspose(transb), m,
        n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC,
        batchCount));
  }

private:
  alpaka::QueueCudaRtNonBlocking m_queue;

  static std::pair<std::size_t, std::size_t>
  layoutKeyA(char trans, std::size_t m, std::size_t k) {
    return (trans == 'N' || trans == 'n') ? std::make_pair(m, k)
                                          : std::make_pair(k, m);
  }

  static std::pair<std::size_t, std::size_t>
  layoutKeyB(char trans, std::size_t k, std::size_t n) {
    return (trans == 'N' || trans == 'n') ? std::make_pair(k, n)
                                          : std::make_pair(n, k);
  }

  // Sets a role's layout descriptor to the given physical (rows, cols),
  // creating it on first use. Matrices are dense column-major, so ld = rows.
  cublasLtMatrixLayout_t
  stampLayout(LayoutRole role, const std::pair<std::size_t, std::size_t> &key) {
    const uint64_t rows = key.first, cols = key.second;
    const int64_t ld = static_cast<int64_t>(key.first);
    cublasLtMatrixLayout_t &L = roleLayout[role];
    if (!L) {
      CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&L, CUDA_R_32F, rows, cols, ld));
    } else {
      CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          L, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows)));
      CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          L, CUBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols)));
      CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          L, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));
    }
    return L;
  }

  // Returns the shape declared through addLayoutConfig that this call belongs
  // to, or null if none covers it. Shapes are the physical (rows, cols) of
  // matrices A, B and C, after any transpose is applied (transa='T' makes
  // shapeA = (k, m)). The contraction dimension (colsA / rowsB) must match
  // exactly: it comes from the weight tensor and never varies at runtime, so
  // it identifies the call site and stops one site's declared shape from
  // serving another's calls. The free dimensions only need covering; among
  // candidates the least excess wins.
  const ShapeEnvelope *
  findEnvelope(const std::pair<std::size_t, std::size_t> &shapeA,
               const std::pair<std::size_t, std::size_t> &shapeB,
               const std::pair<std::size_t, std::size_t> &shapeC) const {
    const ShapeEnvelope *best = nullptr;
    std::size_t bestExcess = std::numeric_limits<std::size_t>::max();
    for (const auto &e : envelopes) {
      if (e.colsA != shapeA.second || e.rowsB != shapeB.first)
        continue;
      if (e.rowsA < shapeA.first || e.colsB < shapeB.second ||
          e.rowsC < shapeC.first || e.colsC < shapeC.second)
        continue;
      const std::size_t ex =
          (e.rowsA - shapeA.first) + (e.colsA - shapeA.second) +
          (e.rowsB - shapeB.first) + (e.colsB - shapeB.second);
      if (ex < bestExcess) {
        bestExcess = ex;
        best = &e;
      }
    }
    return best;
  }

  cublasLtMatmulDesc_t &getOrCreateDesc(cublasOperation_t transA,
                                        cublasOperation_t transB,
                                        cublasLtEpilogue_t epilogue) {
    DescKey key{(int)transA, (int)transB, (int)epilogue};
    auto it = descStore.find(key);
    if (it != descStore.end())
      return it->second;

    cublasLtMatmulDesc_t desc = nullptr;
    CHECK_CUBLAS(
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    // For bias epilogues: set a non-null dummy pointer so the descriptor is
    // valid for cublasLtMatmulAlgoGetHeuristic.
    if (epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
      const void *dummy = d_workspace;
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dummy, sizeof(dummy)));
    }
    descStore.emplace(key, desc);
    return descStore.at(key);
  }

  // Whether the given algorithm can run this exact shape within the
  // workspace. An algorithm resolved at a declared shape is not guaranteed to
  // run at every smaller size it covers.
  bool algoUsable(cublasLtMatmulDesc_t desc, const cublasLtMatmulAlgo_t &algo,
                  const std::pair<std::size_t, std::size_t> &shapeA,
                  const std::pair<std::size_t, std::size_t> &shapeB,
                  const std::pair<std::size_t, std::size_t> &shapeC) {
    auto lA = stampLayout(ROLE_A, shapeA);
    auto lB = stampLayout(ROLE_B, shapeB);
    auto lC = stampLayout(ROLE_C, shapeC);
    cublasLtMatmulHeuristicResult_t chk{};
    return cublasLtMatmulAlgoCheck(ltHandle, desc, lA, lB, lC, lC, &algo,
                                   &chk) == CUBLAS_STATUS_SUCCESS &&
           chk.workspaceSize <= workspaceSize;
  }

  // Looks up, or resolves and caches, the algorithm for the given transpose
  // settings, epilogue and shapes. required=false is for constructor warmup:
  // a speculatively resolved epilogue may legitimately have no algorithm, and
  // returns null instead of aborting.
  cublasLtMatmulHeuristicResult_t *
  getOrComputeAlgo(cublasOperation_t transA, cublasOperation_t transB,
                   cublasLtEpilogue_t epilogue,
                   const std::pair<std::size_t, std::size_t> &shapeA,
                   const std::pair<std::size_t, std::size_t> &shapeB,
                   const std::pair<std::size_t, std::size_t> &shapeC,
                   bool required = true) {
    AlgoKey key{{(int)transA, (int)transB, (int)epilogue},
                shapeA.first,
                shapeA.second,
                shapeB.first,
                shapeB.second};
    auto it = algoCache.find(key);
    if (it != algoCache.end()) {
      if (algoCacheLimit)
        lruOrder.splice(lruOrder.begin(), lruOrder, it->second.lru);
      return &it->second.h;
    }

    auto &desc = getOrCreateDesc(transA, transB, epilogue);
    auto lA = stampLayout(ROLE_A, shapeA);
    auto lB = stampLayout(ROLE_B, shapeB);
    auto lC = stampLayout(ROLE_C, shapeC);
    cublasLtMatmulHeuristicResult_t h{};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, desc, lA, lB, lC, lC, preference, 1, &h, &returnedResults));
    ++stats.heuristicQueries;
    if (returnedResults == 0) {
      if (!required)
        return nullptr;
      std::cerr << "[sofieBLAS] No suitable cuBLASLt algorithm found for "
                << "transA=" << transA << " transB=" << transB
                << " epilogue=" << epilogue << " A=[" << shapeA.first << "x"
                << shapeA.second << "]"
                << " B=[" << shapeB.first << "x" << shapeB.second << "]\n";
      exit(EXIT_FAILURE);
    }
    auto ins = algoCache.emplace(key, CacheEntry{h, {}}).first;
    if (algoCacheLimit) {
      lruOrder.push_front(key);
      ins->second.lru = lruOrder.begin();
      while (algoCache.size() > algoCacheLimit) {
        algoCache.erase(lruOrder.back());
        lruOrder.pop_back();
        ++stats.evictions;
      }
    }
    return &ins->second.h;
  }

  void executeMatmul(cublasOperation_t transA, cublasOperation_t transB,
                     cublasLtEpilogue_t epilogue, float alpha, const float *A,
                     const float *B, float beta, const float *D_in,
                     float *C_out, const void *bias_ptr,
                     const std::pair<std::size_t, std::size_t> &shapeA,
                     const std::pair<std::size_t, std::size_t> &shapeB,
                     const std::pair<std::size_t, std::size_t> &shapeC) {
    auto &desc = getOrCreateDesc(transA, transB, epilogue);
    if (bias_ptr) {
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
          sizeof(bias_ptr)));
    }

    // Resolve the algorithm at the call site's declared shape when one covers
    // this call, so every runtime size the site produces shares one cache
    // entry; with no covering declaration, resolve at the exact shape.
    const ShapeEnvelope *env = findEnvelope(shapeA, shapeB, shapeC);
    const std::pair<std::size_t, std::size_t>
        algoShapeA = env ? std::make_pair(env->rowsA, env->colsA) : shapeA,
        algoShapeB = env ? std::make_pair(env->rowsB, env->colsB) : shapeB,
        algoShapeC = env ? std::make_pair(env->rowsC, env->colsC) : shapeC;
    cublasLtMatmulHeuristicResult_t h = *getOrComputeAlgo(
        transA, transB, epilogue, algoShapeA, algoShapeB, algoShapeC);

    // Normally the resolution above is the only one. Only when cuBLASLt
    // rejects the declared shape's algorithm at this call's exact size
    // (returns NOT_SUPPORTED; m=1 was found to do this in testing) is the
    // algorithm resolved a second time, at the exact shape, and that result
    // is cached as well.
    if (env && !algoUsable(desc, h.algo, shapeA, shapeB, shapeC)) {
      ++stats.envelopeRejects;
      h = *getOrComputeAlgo(transA, transB, epilogue, shapeA, shapeB, shapeC);
    }

    // Stamp the exact call shape last: the algorithm resolution and the
    // validity check above leave the shared descriptors at other dims.
    auto lA = stampLayout(ROLE_A, shapeA);
    auto lB = stampLayout(ROLE_B, shapeB);
    auto lC = stampLayout(ROLE_C, shapeC);
    CHECK_CUBLAS(cublasLtMatmul(ltHandle, desc, &alpha, A, lA, B, lB, &beta,
                                D_in, lC, C_out, lC, &h.algo, d_workspace,
                                workspaceSize, stream));
  }
};

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuCudaRt> {
public:
  using Impl = BlasCuda;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
