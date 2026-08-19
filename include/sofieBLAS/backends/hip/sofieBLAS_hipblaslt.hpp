#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

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
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#define CHECK_HIP(err)                                                         \
  if ((err) != hipSuccess) {                                                   \
    std::cerr << "HIP error: " << hipGetErrorString(err) << " at line "        \
              << __LINE__ << "\n";                                             \
    exit(EXIT_FAILURE);                                                        \
  }

#define CHECK_HIPBLAS(status)                                                  \
  do {                                                                         \
    hipblasStatus_t _s = (status);                                             \
    if (_s != HIPBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "hipBLAS error " << _s << " at line " << __LINE__ << "\n";  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

struct DescKey {
  int transA; // HIPBLAS_OP_N / HIPBLAS_OP_T encoded as int
  int transB;
  int epilogue; // hipblasLtEpilogue_t encoded as int
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

struct ShapeEnvelope {
  std::size_t rowsA, colsA, rowsB, colsB, rowsC, colsC;
};

struct LayoutStats {
  std::size_t heuristicQueries = 0;
  std::size_t envelopeRejects = 0;
  std::size_t evictions = 0;
};

class BlasHip {
  hipblasLtHandle_t ltHandle = nullptr;
  hipblasHandle_t handle = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  void *d_workspace = nullptr;
  size_t workspaceSize = 1u << 25; // 32 MB
  hipStream_t stream = nullptr;

  enum LayoutRole { ROLE_A = 0, ROLE_B = 1, ROLE_C = 2 };
  hipblasLtMatrixLayout_t roleLayout[3] = {};

  std::unordered_map<DescKey, hipblasLtMatmulDesc_t, DescKeyHash> descStore;

  // algo cache entry
  struct CacheEntry {
    hipblasLtMatmulHeuristicResult_t h{};
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

  BlasHip(const BlasHip &) = delete;
  BlasHip &operator=(const BlasHip &) = delete;
  BlasHip(BlasHip &&) = delete;
  BlasHip &operator=(BlasHip &&) = delete;

  BlasHip(alpaka::QueueHipRtNonBlocking &queue, std::size_t algoCacheLimit_ = 0)
      : algoCacheLimit{algoCacheLimit_}, m_queue{queue} {
    stream = static_cast<hipStream_t>(m_queue.getNativeHandle());

    CHECK_HIPBLAS(hipblasLtCreate(&ltHandle));

    CHECK_HIPBLAS(hipblasCreate(&handle));
    CHECK_HIPBLAS(hipblasSetStream(handle, stream));

    CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&preference));
    CHECK_HIP(hipMalloc(&d_workspace, workspaceSize));
    CHECK_HIPBLAS(hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }

  ~BlasHip() {
    for (auto L : roleLayout)
      if (L)
        hipblasLtMatrixLayoutDestroy(L);
    for (auto &[key, desc] : descStore)
      if (desc)
        hipblasLtMatmulDescDestroy(desc);
    if (preference)
      hipblasLtMatmulPreferenceDestroy(preference);
    if (ltHandle)
      hipblasLtDestroy(ltHandle);
    if (handle)
      hipblasDestroy(handle);
    if (d_workspace)
      hipFree(d_workspace);
  }

  inline hipblasOperation_t charToHipBlasTranspose(char trans) {
    switch (trans) {
    case 'N':
    case 'n':
      return HIPBLAS_OP_N;
    case 'T':
    case 't':
      return HIPBLAS_OP_T;
    case 'C':
    case 'c':
      return HIPBLAS_OP_C;
    default:
      throw std::invalid_argument("Invalid transpose character for hipBLAS.");
    }
  }

  void addLayoutConfig(std::size_t m, std::size_t n, std::size_t k, std::size_t,
                       std::size_t, std::size_t, char transa, char transb) {
    const auto kA = layoutKeyA(transa, m, k);
    const auto kB = layoutKeyB(transb, k, n);
    const std::pair<std::size_t, std::size_t> kC{m, n};
    envelopes.push_back({kA.first, kA.second, kB.first, kB.second, m, n});

    const hipblasOperation_t tA = charToHipBlasTranspose(transa);
    const hipblasOperation_t tB = charToHipBlasTranspose(transb);
    const hipblasLtEpilogue_t eps[] = {HIPBLASLT_EPILOGUE_DEFAULT,
                                       HIPBLASLT_EPILOGUE_BIAS,
                                       HIPBLASLT_EPILOGUE_RELU_BIAS};
    for (hipblasLtEpilogue_t ep : eps) {
      getOrComputeAlgo(tA, tB, ep, kA, kB, kC, /*required=*/false);
    }
  }

  template <typename T, typename TIdx>
  inline void
  gemm(char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
       float alpha, alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> const &A,
       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> const &B, float beta,
       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> &bias,
       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_BIAS,
                  alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemm(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> const
          &A,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> const
          &B,
      float beta,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> &bias,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_BIAS,
                  alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, T const *B,
                   float beta, T *bias, T *C) {
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_BIAS,
                  alpha, A, B, beta, bias, C, static_cast<const void *>(bias),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_RELU_BIAS,
                  alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmrelu(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> const
          &A,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> const
          &B,
      float beta,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> &bias,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_RELU_BIAS,
                  alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_RELU_BIAS,
                  alpha, A, B, beta, bias, C, static_cast<const void *>(bias),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_GELU_BIAS,
                  alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmgelu(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> const
          &A,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> const
          &B,
      float beta,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> &bias,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_GELU_BIAS,
                  alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_GELU_BIAS,
                  alpha, A, B, beta, bias, C, static_cast<const void *>(bias),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha,
                     alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                     alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                     float beta,
                     alpaka::BufHipRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    float *c = alpaka::getPtrNative(C);
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_DEFAULT,
                  alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  c, c, nullptr, layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void matmul(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> const
          &A,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> const
          &B,
      float beta,
      alpaka::ViewPlainPtr<alpaka::DevHipRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    T *c = alpaka::getPtrNative(C);
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_DEFAULT,
                  alpha, alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  c, c, nullptr, layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, T const *B,
                     float beta, T *C) {
    executeMatmul(charToHipBlasTranspose(transa),
                  charToHipBlasTranspose(transb), HIPBLASLT_EPILOGUE_DEFAULT,
                  alpha, A, B, beta, C, C, nullptr, layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  inline void gemmStridedBatched(char transa, char transb, int m, int n, int k,
                                 float alpha, const float *A, int lda,
                                 long long strideA, const float *B, int ldb,
                                 long long strideB, float beta, float *C,
                                 int ldc, long long strideC, int batchCount) {
    CHECK_HIPBLAS(hipblasSgemmStridedBatched(
        handle, charToHipBlasTranspose(transa), charToHipBlasTranspose(transb),
        m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc,
        strideC, batchCount));
  }

private:
  alpaka::QueueHipRtNonBlocking m_queue;

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

  hipblasLtMatrixLayout_t
  stampLayout(LayoutRole role, const std::pair<std::size_t, std::size_t> &key) {
    const uint64_t rows = key.first, cols = key.second;
    const int64_t ld = static_cast<int64_t>(key.first);
    hipblasLtMatrixLayout_t &L = roleLayout[role];
    if (!L) {
      CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&L, HIP_R_32F, rows, cols, ld));
    } else {
      CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(
          L, HIPBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows)));
      CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(
          L, HIPBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols)));
      CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(
          L, HIPBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));
    }
    return L;
  }

  const ShapeEnvelope *
  findEnvelope(const std::pair<std::size_t, std::size_t> &kA,
               const std::pair<std::size_t, std::size_t> &kB,
               const std::pair<std::size_t, std::size_t> &kC) const {
    const ShapeEnvelope *best = nullptr;
    std::size_t bestExcess = std::numeric_limits<std::size_t>::max();
    for (const auto &e : envelopes) {
      if (e.colsA != kA.second || e.rowsB != kB.first)
        continue;
      if (e.rowsA < kA.first || e.colsB < kB.second || e.rowsC < kC.first ||
          e.colsC < kC.second)
        continue;
      const std::size_t ex = (e.rowsA - kA.first) + (e.colsA - kA.second) +
                             (e.rowsB - kB.first) + (e.colsB - kB.second);
      if (ex < bestExcess) {
        bestExcess = ex;
        best = &e;
      }
    }
    return best;
  }

  bool algoUsable(hipblasLtMatmulDesc_t desc, const hipblasLtMatmulAlgo_t &algo,
                  const std::pair<std::size_t, std::size_t> &kA,
                  const std::pair<std::size_t, std::size_t> &kB,
                  const std::pair<std::size_t, std::size_t> &kC) {
    auto lA = stampLayout(ROLE_A, kA);
    auto lB = stampLayout(ROLE_B, kB);
    auto lC = stampLayout(ROLE_C, kC);
    // hipBLASLt has no hipblasLtMatmulAlgoCheck; the ext API rewrites the algo
    // it is handed, so probe a copy.
    hipblasLtMatmulAlgo_t probe = algo;
    const float alpha = 1.f, beta = 0.f;
    std::size_t ws = 0;
    return hipblaslt_ext::matmulIsAlgoSupported(ltHandle, desc, &alpha, lA, lB,
                                                &beta, lC, lC, probe,
                                                ws) == HIPBLAS_STATUS_SUCCESS &&
           ws <= workspaceSize;
  }

  hipblasLtMatmulDesc_t &getOrCreateDesc(hipblasOperation_t transA,
                                         hipblasOperation_t transB,
                                         hipblasLtEpilogue_t epilogue) {
    DescKey key{(int)transA, (int)transB, (int)epilogue};
    auto it = descStore.find(key);
    if (it != descStore.end())
      return it->second;

    hipblasLtMatmulDesc_t desc = nullptr;
    CHECK_HIPBLAS(
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (epilogue != HIPBLASLT_EPILOGUE_DEFAULT) {
      const void *dummy = d_workspace;
      CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
          desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dummy, sizeof(dummy)));
    }
    descStore.emplace(key, desc);
    return descStore.at(key);
  }

  hipblasLtMatmulHeuristicResult_t *
  getOrComputeAlgo(hipblasOperation_t transA, hipblasOperation_t transB,
                   hipblasLtEpilogue_t epilogue,
                   const std::pair<std::size_t, std::size_t> &kA,
                   const std::pair<std::size_t, std::size_t> &kB,
                   const std::pair<std::size_t, std::size_t> &kC,
                   bool required = true) {
    AlgoKey key{{(int)transA, (int)transB, (int)epilogue},
                kA.first,
                kA.second,
                kB.first,
                kB.second};
    auto it = algoCache.find(key);
    if (it != algoCache.end()) {
      if (algoCacheLimit)
        lruOrder.splice(lruOrder.begin(), lruOrder, it->second.lru);
      return &it->second.h;
    }

    auto &desc = getOrCreateDesc(transA, transB, epilogue);
    auto lA = stampLayout(ROLE_A, kA);
    auto lB = stampLayout(ROLE_B, kB);
    auto lC = stampLayout(ROLE_C, kC);
    hipblasLtMatmulHeuristicResult_t h{};
    int returnedResults = 0;
    CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(
        ltHandle, desc, lA, lB, lC, lC, preference, 1, &h, &returnedResults));
    ++stats.heuristicQueries;
    if (returnedResults == 0) {
      if (!required)
        return nullptr;
      std::cerr << "[sofieBLAS] No suitable hipBLASLt algorithm found for "
                << "transA=" << transA << " transB=" << transB
                << " epilogue=" << epilogue << " A=[" << kA.first << "x"
                << kA.second << "]"
                << " B=[" << kB.first << "x" << kB.second << "]\n";
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

  void executeMatmul(hipblasOperation_t transA, hipblasOperation_t transB,
                     hipblasLtEpilogue_t epilogue, float alpha, const float *A,
                     const float *B, float beta, const float *D_in,
                     float *C_out, const void *bias_ptr,
                     const std::pair<std::size_t, std::size_t> &kA,
                     const std::pair<std::size_t, std::size_t> &kB,
                     const std::pair<std::size_t, std::size_t> &kC) {
    auto &desc = getOrCreateDesc(transA, transB, epilogue);
    if (bias_ptr) {
      CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
          desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
          sizeof(bias_ptr)));
    }

    const ShapeEnvelope *env = findEnvelope(kA, kB, kC);
    const std::pair<std::size_t, std::size_t>
        aA = env ? std::make_pair(env->rowsA, env->colsA) : kA,
        aB = env ? std::make_pair(env->rowsB, env->colsB) : kB,
        aC = env ? std::make_pair(env->rowsC, env->colsC) : kC;
    hipblasLtMatmulHeuristicResult_t h =
        *getOrComputeAlgo(transA, transB, epilogue, aA, aB, aC);

    if (env && !algoUsable(desc, h.algo, kA, kB, kC)) {
      ++stats.envelopeRejects;
      h = *getOrComputeAlgo(transA, transB, epilogue, kA, kB, kC);
    }

    auto lA = stampLayout(ROLE_A, kA);
    auto lB = stampLayout(ROLE_B, kB);
    auto lC = stampLayout(ROLE_C, kC);
    CHECK_HIPBLAS(hipblasLtMatmul(ltHandle, desc, &alpha, A, lA, B, lB, &beta,
                                  D_in, lC, C_out, lC, &h.algo, d_workspace,
                                  workspaceSize, stream));
  }
};

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuHipRt> {
public:
  using Impl = BlasHip;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
