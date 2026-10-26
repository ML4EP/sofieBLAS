#pragma once

// Helpers behind the GPU backends' public API
// the cache keys, and BlasLtBase, the part of the cuBLASLt / hipBLASLt wrapper
// that owns the handles and caches, builds layouts and descriptors, resolves
// algorithms and runs the multiplies. The Api table and the check macros
// SOFIEBLAS_CHECK_LT / SOFIEBLAS_CHECK_RT come from the vendor header, which
// includes this file after defining them.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "sofieBLAS/core.hpp"

struct PairHash {
  std::size_t
  operator()(const std::pair<std::size_t, std::size_t> &p) const noexcept {
    std::size_t h1 = std::hash<std::size_t>{}(p.first);
    std::size_t h2 = std::hash<std::size_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct LayoutKey {
  int type;
  std::size_t rows, cols;
  bool operator==(const LayoutKey &o) const noexcept {
    return type == o.type && rows == o.rows && cols == o.cols;
  }
};

struct LayoutKeyHash {
  std::size_t operator()(const LayoutKey &k) const noexcept {
    std::size_t h = PairHash{}({k.rows, k.cols});
    return h ^ (std::hash<int>{}(k.type) + 0x9e3779b97f4a7c15ULL + (h << 6) +
                (h >> 2));
  }
};

struct LdKey {
  int type;
  std::size_t rows, cols, ld;
  bool operator==(const LdKey &o) const noexcept {
    return type == o.type && rows == o.rows && cols == o.cols && ld == o.ld;
  }
};

struct LdKeyHash {
  std::size_t operator()(const LdKey &k) const noexcept {
    std::size_t h =
        PairHash{}({k.rows, k.cols}) ^ (std::hash<std::size_t>{}(k.ld) * 31u);
    return h ^ (std::hash<int>{}(k.type) + 0x9e3779b97f4a7c15ULL + (h << 6) +
                (h >> 2));
  }
};

struct DescKey {
  int transA; // backend transpose enum encoded as int
  int transB;
  int epilogue;        // backend epilogue enum encoded as int
  int typeD;           // backend data-type enum of the output D
  int alphaVector = 0; // 1 when alpha is a device vector, one value per row
  bool operator==(const DescKey &o) const noexcept {
    return transA == o.transA && transB == o.transB && epilogue == o.epilogue &&
           typeD == o.typeD && alphaVector == o.alphaVector;
  }
};

struct DescKeyHash {
  std::size_t operator()(const DescKey &k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.transA) * 97u +
                    static_cast<std::size_t>(k.transB) * 31u +
                    static_cast<std::size_t>(k.epilogue);
    h = h * 31u + static_cast<std::size_t>(k.typeD);
    h = h * 31u + static_cast<std::size_t>(k.alphaVector);
    return h ^ (h >> 16);
  }
};

struct AlgoKey {
  DescKey dk;
  std::size_t rowsA, colsA;     // physical dimensions of A in layoutStore
  std::size_t rowsB, colsB;     // physical dimensions of B in layoutStore
  std::size_t ldA = 0, ldB = 0; // leading dimensions of A and B, 0 when dense
  bool operator==(const AlgoKey &o) const noexcept {
    return dk == o.dk && rowsA == o.rowsA && colsA == o.colsA &&
           rowsB == o.rowsB && colsB == o.colsB && ldA == o.ldA && ldB == o.ldB;
  }
};

struct AlgoKeyHash {
  std::size_t operator()(const AlgoKey &k) const noexcept {
    std::size_t h = DescKeyHash{}(k.dk);
    // folds the hash of one field into the running hash h, so that keys
    // differing in any field (or in field order) hash differently
    auto combineHash = [&](std::size_t v) {
      h ^= std::hash<std::size_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) +
           (h >> 2);
    };
    combineHash(k.rowsA);
    combineHash(k.colsA);
    combineHash(k.rowsB);
    combineHash(k.colsB);
    combineHash(k.ldA);
    combineHash(k.ldB);
    return h;
  }
};

template <class Api> class BlasLtBase {
protected:
  typename Api::Handle ltHandle = nullptr;
  typename Api::BlasHandle handle = nullptr;
  typename Api::Preference preference = nullptr;
  void *d_workspace = nullptr;
  size_t workspaceSize = 1u << 25; // 32 MB
  typename Api::Stream stream = nullptr;

  std::unordered_map<LayoutKey, typename Api::Layout, LayoutKeyHash>
      layoutStore;

  // layouts of the operands with a leading dimension larger than the number of
  // rows (padded or sub-matrix views); the dense ones are in layoutStore
  std::unordered_map<LdKey, typename Api::Layout, LdKeyHash> ldLayoutStore;

  // One cache entry per exact GEMM configuration: the heuristic result to
  // reuse, plus this entry's position in the recency list so a hit can mark
  // itself most-recently-used in O(1). The position is only maintained when a
  // cache limit is set; with no limit the list stays empty.
  struct CacheEntry {
    typename Api::HeuristicResult h{};
    std::list<AlgoKey>::iterator lru{};
  };

  // The descriptors and algorithms of one group of input types. Each group
  // has its own set, so two groups never share a descriptor (or what is set on
  // it per call) or an algorithm. Inside a group the output type decides every
  // other type, so it is the only type in the keys. Layouts are shared by all
  // groups: their key holds the element type and nothing is set on them per
  // call.
  struct GemmCaches {
    std::unordered_map<DescKey, typename Api::MatmulDesc, DescKeyHash>
        descStore;
    std::unordered_map<AlgoKey, CacheEntry, AlgoKeyHash> algoCache;
    // entries ordered most- to least-recently used; drives eviction
    std::list<AlgoKey> lruOrder;
  };
  GemmCaches floatCaches, fp16Caches, int8Caches, fp8Caches;
  // 0 = unbounded; the limit applies to each group
  std::size_t algoCacheLimit = 0;

public:
  BlasLtBase(const BlasLtBase &) = delete;
  BlasLtBase &operator=(const BlasLtBase &) = delete;
  BlasLtBase(BlasLtBase &&) = delete;
  BlasLtBase &operator=(BlasLtBase &&) = delete;

  BlasLtBase(typename Api::Queue &queue, std::size_t cacheLimit = 0)
      : algoCacheLimit{cacheLimit}, m_queue{queue} {
    stream = static_cast<typename Api::Stream>(m_queue.getNativeHandle());

    SOFIEBLAS_CHECK_LT(Api::ltCreate(&ltHandle));

    SOFIEBLAS_CHECK_LT(Api::blasCreate(&handle));
    SOFIEBLAS_CHECK_LT(Api::blasSetStream(handle, stream));

    SOFIEBLAS_CHECK_LT(Api::prefCreate(&preference));
    SOFIEBLAS_CHECK_RT(Api::rtMalloc(&d_workspace, workspaceSize));
    SOFIEBLAS_CHECK_LT(Api::prefSetAttribute(preference, Api::PrefMaxWorkspace,
                                             &workspaceSize,
                                             sizeof(workspaceSize)));
  }

  ~BlasLtBase() {
    for (auto &[key, layout] : layoutStore)
      if (layout)
        Api::layoutDestroy(layout);
    for (auto &[key, layout] : ldLayoutStore)
      if (layout)
        Api::layoutDestroy(layout);
    for (auto &[key, entry] : batchedStore) {
      Api::layoutDestroy(entry.lA);
      Api::layoutDestroy(entry.lB);
      Api::layoutDestroy(entry.lC);
    }
    for (auto &[key, entry] : quantBatchedStore)
      destroyQuantBatched(entry);
    for (auto *caches : {&floatCaches, &fp16Caches, &int8Caches, &fp8Caches})
      for (auto &[key, desc] : caches->descStore)
        if (desc)
          Api::descDestroy(desc);
    if (preference)
      Api::prefDestroy(preference);
    if (ltHandle)
      Api::ltDestroy(ltHandle);
    if (handle)
      Api::blasDestroy(handle);
    if (d_workspace)
      Api::rtFree(d_workspace);
  }

protected:
  inline typename Api::Operation charToTranspose(char trans) {
    switch (trans) {
    case 'N':
    case 'n':
      return Api::OpN;
    case 'T':
    case 't':
      return Api::OpT;
    case 'C':
    case 'c':
      return Api::OpC;
    default:
      throw std::invalid_argument(
          std::string("Invalid transpose character for ") + Api::name + ".");
    }
  }

  typename Api::Queue m_queue;

  static typename Api::Epilogue toApiEpilogue(Epilogue epilogue) {
    switch (epilogue) {
    case Epilogue::Bias:
      return Api::EpilogueBias;
    case Epilogue::ReluBias:
      return Api::EpilogueReluBias;
    case Epilogue::GeluBias:
      return Api::EpilogueGeluBias;
    case Epilogue::Relu:
      return Api::EpilogueRelu;
    case Epilogue::Default:
      break;
    }
    return Api::EpilogueDefault;
  }

  // true for the epilogues that add a bias vector
  static bool usesBias(typename Api::Epilogue epilogue) {
    return epilogue != Api::EpilogueDefault && epilogue != Api::EpilogueRelu;
  }

  // the cache set of the group a GEMM belongs to, chosen by the type of A and B
  GemmCaches &cachesFor(decltype(Api::RealF32) typeAB) {
    if (typeAB == Api::RealF32)
      return floatCaches;
    if (typeAB == Api::RealF16)
      return fp16Caches;
    if (typeAB == Api::RealI8)
      return int8Caches;
    if (typeAB == Api::RealF8E4M3)
      return fp8Caches;
    throw std::invalid_argument("sofieBLAS: no cache set for this input type");
  }

  static bool isFp8(decltype(Api::RealF32) type) {
    return type == Api::RealF8E4M3 || type == Api::RealF8E5M2;
  }

  // the type of C: the library needs a 16-bit C when D is FP8 (beta is 0 for
  // those, so C is never read); otherwise C has D's type
  static decltype(Api::RealF32) cTypeFor(decltype(Api::RealF32) typeD) {
    return isFp8(typeD) ? Api::RealBF16 : typeD;
  }

  // the configuration of a batched multiply: transposes, epilogue, m, n, k,
  // the leading dimensions, the three strides and the batch count
  using BatchKey = std::array<long long, 13>;
  struct BatchedEntry {
    typename Api::Layout lA = nullptr, lB = nullptr, lC = nullptr;
    typename Api::HeuristicResult h{};
  };
  std::map<BatchKey, BatchedEntry> batchedStore;

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

  // the configuration of a quantized batched multiply (dense matrices):
  // transposes, epilogue, m, n, k, the three strides, the batch count and the
  // types of A and B and of D
  using QuantBatchKey = std::array<long long, 12>;
  struct QuantBatchedEntry {
    typename Api::Layout lA = nullptr, lB = nullptr, lC = nullptr, lD = nullptr;
    typename Api::HeuristicResult h{};
  };
  std::map<QuantBatchKey, QuantBatchedEntry> quantBatchedStore;

  typename Api::Layout
  createBatchedLayout(const std::pair<std::size_t, std::size_t> &shape,
                      std::size_t ld, long long stride, int batchCount,
                      decltype(Api::RealF32) type = Api::RealF32) {
    typename Api::Layout layout = nullptr;
    SOFIEBLAS_CHECK_LT(
        Api::layoutCreate(&layout, type, shape.first, shape.second, ld));
    std::int32_t count = batchCount;
    std::int64_t offset = stride;
    SOFIEBLAS_CHECK_LT(Api::layoutSetAttribute(layout, Api::LayoutBatchCount,
                                               &count, sizeof(count)));
    SOFIEBLAS_CHECK_LT(Api::layoutSetAttribute(
        layout, Api::LayoutStridedBatchOffset, &offset, sizeof(offset)));
    return layout;
  }

  static void destroyQuantBatched(QuantBatchedEntry &e) {
    if (e.lD && e.lD != e.lC)
      Api::layoutDestroy(e.lD);
    for (auto layout : {e.lA, e.lB, e.lC})
      if (layout)
        Api::layoutDestroy(layout);
  }

  // Returns the cached layouts and algorithm of a quantized batched multiply,
  // computing them on first use. Without a kernel it ends the process.
  QuantBatchedEntry *getOrComputeQuantBatched(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      long long strideA, long long strideB, long long strideC, int batchCount,
      Epilogue epilogue, decltype(Api::ComputeF32) computeType,
      decltype(Api::RealF32) scaleType, decltype(Api::RealF32) typeAB,
      decltype(Api::RealF32) typeD, decltype(Api::RealF32) biasType) {
    if (batchCount < 1)
      throw std::invalid_argument("sofieBLAS: batchCount must be positive.");
    const auto opA = charToTranspose(transa);
    const auto opB = charToTranspose(transb);
    const auto epi = toApiEpilogue(epilogue);
    QuantBatchKey key{(long long)opA,    (long long)opB,
                      (long long)epi,    (long long)m,
                      (long long)n,      (long long)k,
                      strideA,           strideB,
                      strideC,           (long long)batchCount,
                      (long long)typeAB, (long long)typeD};
    auto it = quantBatchedStore.find(key);
    if (it != quantBatchedStore.end())
      return &it->second;

    const auto shapeA = layoutKeyA(transa, m, k);
    const auto shapeB = layoutKeyB(transb, k, n);
    const std::pair<std::size_t, std::size_t> shapeC{m, n};
    auto &desc = getOrCreateDesc(opA, opB, epi, computeType, scaleType,
                                 biasType, typeAB, typeD);
    QuantBatchedEntry e{};
    e.lA =
        createBatchedLayout(shapeA, shapeA.first, strideA, batchCount, typeAB);
    e.lB =
        createBatchedLayout(shapeB, shapeB.first, strideB, batchCount, typeAB);
    e.lC = createBatchedLayout(shapeC, m, strideC, batchCount, cTypeFor(typeD));
    e.lD = cTypeFor(typeD) == typeD
               ? e.lC
               : createBatchedLayout(shapeC, m, strideC, batchCount, typeD);
    int returned = 0;
    const auto status = Api::getHeuristic(ltHandle, desc, e.lA, e.lB, e.lC,
                                          e.lD, preference, 1, &e.h, &returned);
    SOFIEBLAS_CHECK_LT(status);
    if (returned == 0) {
      std::cerr << "[sofieBLAS] No suitable " << Api::name
                << " algorithm found for the quantized batched multiply m=" << m
                << " n=" << n << " k=" << k << " batchCount=" << batchCount
                << " strideA=" << strideA << " strideB=" << strideB << "\n";
      exit(EXIT_FAILURE);
    }
    return &quantBatchedStore.emplace(key, e).first->second;
  }

  // Returns the layout describing a (rows, cols) matrix of the given element
  // type with leading dimension ld (column-major), creating and caching it on
  // first use. The dense layouts (ld = rows) are keyed by the type and shape,
  // the others also by ld.
  typename Api::Layout
  getOrCreateLayout(const std::pair<std::size_t, std::size_t> &shape,
                    std::size_t ld,
                    decltype(Api::RealF32) type = Api::RealF32) {
    if (ld < shape.first)
      throw std::invalid_argument(
          std::string("sofieBLAS: leading dimension ") + std::to_string(ld) +
          " is smaller than the number of rows " + std::to_string(shape.first) +
          " of the matrix.");
    if (ld == shape.first) {
      const LayoutKey key{static_cast<int>(type), shape.first, shape.second};
      auto it = layoutStore.find(key);
      if (it != layoutStore.end())
        return it->second;
      typename Api::Layout layout = nullptr;
      SOFIEBLAS_CHECK_LT(
          Api::layoutCreate(&layout, type, shape.first, shape.second, ld));
      layoutStore.emplace(key, layout);
      return layout;
    }
    LdKey key{static_cast<int>(type), shape.first, shape.second, ld};
    auto it = ldLayoutStore.find(key);
    if (it != ldLayoutStore.end())
      return it->second;
    typename Api::Layout layout = nullptr;
    SOFIEBLAS_CHECK_LT(
        Api::layoutCreate(&layout, type, shape.first, shape.second, ld));
    ldLayoutStore.emplace(key, layout);
    return layout;
  }

  typename Api::MatmulDesc &getOrCreateDesc(
      typename Api::Operation transA, typename Api::Operation transB,
      typename Api::Epilogue epilogue,
      decltype(Api::ComputeF32) computeType = Api::ComputeF32,
      decltype(Api::RealF32) scaleType = Api::RealF32,
      decltype(Api::RealF32) biasType = Api::RealF32,
      decltype(Api::RealF32) typeAB = Api::RealF32,
      decltype(Api::RealF32) typeD = Api::RealF32, bool alphaVector = false) {
    auto &descStore = cachesFor(typeAB).descStore;
    DescKey key{(int)transA, (int)transB, (int)epilogue, (int)typeD,
                alphaVector ? 1 : 0};
    auto it = descStore.find(key);
    if (it != descStore.end())
      return it->second;

    typename Api::MatmulDesc desc = nullptr;
    SOFIEBLAS_CHECK_LT(Api::descCreate(&desc, computeType, scaleType));
    SOFIEBLAS_CHECK_LT(
        Api::descSetAttribute(desc, Api::DescTransA, &transA, sizeof(transA)));
    SOFIEBLAS_CHECK_LT(
        Api::descSetAttribute(desc, Api::DescTransB, &transB, sizeof(transB)));
    SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescEpilogue, &epilogue,
                                             sizeof(epilogue)));
    // For bias epilogues: set a non-null dummy pointer so the descriptor is
    // valid for the heuristic query, and the bias type the caller declared.
    if (usesBias(epilogue)) {
      const void *dummy = d_workspace;
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasPointer,
                                               &dummy, sizeof(dummy)));
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasDataType,
                                               &biasType, sizeof(biasType)));
    }
    // FP8: placeholder scale pointers, so the algorithm search sees a scaled
    // multiply; the real ones are set on every call
    if (typeAB == Api::RealF8E4M3) {
      const void *dummy = d_workspace;
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescAScalePointer,
                                               &dummy, sizeof(dummy)));
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBScalePointer,
                                               &dummy, sizeof(dummy)));
      if (isFp8(typeD))
        SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescDScalePointer,
                                                 &dummy, sizeof(dummy)));
    }
    // alpha as a device vector with one value per row of D. Beta is always 0
    // here, but it is passed as a host value: cuBLASLt has column-major int8
    // kernels for the BETA_HOST mode and none for BETA_ZERO (which needs the
    // COL32 layouts of the older int8 kernels)
    if constexpr (requires { Api::PointerModeAlphaDeviceVectorBetaHost; }) {
      if (alphaVector) {
        std::int32_t mode = Api::PointerModeAlphaDeviceVectorBetaHost;
        SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescPointerMode,
                                                 &mode, sizeof(mode)));
      }
    }
    descStore.emplace(key, desc);
    return descStore.at(key);
  }

  typename Api::HeuristicResult *
  getOrComputeAlgo(typename Api::Operation transA,
                   typename Api::Operation transB,
                   typename Api::Epilogue epilogue,
                   const std::pair<std::size_t, std::size_t> &shapeA,
                   const std::pair<std::size_t, std::size_t> &shapeB,
                   const std::pair<std::size_t, std::size_t> &shapeC,
                   std::size_t ldA = 0, std::size_t ldB = 0,
                   decltype(Api::ComputeF32) computeType = Api::ComputeF32,
                   decltype(Api::RealF32) scaleType = Api::RealF32,
                   decltype(Api::RealF32) typeAB = Api::RealF32,
                   decltype(Api::RealF32) typeD = Api::RealF32,
                   decltype(Api::RealF32) biasType = Api::RealF32,
                   bool alphaVector = false) {
    // ldA/ldB are 0 for dense operands
    if (ldA == shapeA.first)
      ldA = 0;
    if (ldB == shapeB.first)
      ldB = 0;
    auto &algoCache = cachesFor(typeAB).algoCache;
    auto &lruOrder = cachesFor(typeAB).lruOrder;
    AlgoKey key{{(int)transA, (int)transB, (int)epilogue, (int)typeD,
                 alphaVector ? 1 : 0},
                shapeA.first,
                shapeA.second,
                shapeB.first,
                shapeB.second,
                ldA,
                ldB};
    auto it = algoCache.find(key);
    if (it != algoCache.end()) {
      if (algoCacheLimit)
        lruOrder.splice(lruOrder.begin(), lruOrder, it->second.lru);
      return &it->second.h;
    }

    auto &desc =
        getOrCreateDesc(transA, transB, epilogue, computeType, scaleType,
                        biasType, typeAB, typeD, alphaVector);
    auto lA = getOrCreateLayout(shapeA, ldA ? ldA : shapeA.first, typeAB);
    auto lB = getOrCreateLayout(shapeB, ldB ? ldB : shapeB.first, typeAB);
    auto lC = getOrCreateLayout(shapeC, shapeC.first, cTypeFor(typeD));
    auto lD = cTypeFor(typeD) == typeD
                  ? lC
                  : getOrCreateLayout(shapeC, shapeC.first, typeD);
    typename Api::HeuristicResult h{};
    int returnedResults = 0;
    const auto status = Api::getHeuristic(ltHandle, desc, lA, lB, lC, lD,
                                          preference, 1, &h, &returnedResults);
    SOFIEBLAS_CHECK_LT(status);
    if (returnedResults == 0) {
      std::cerr << "[sofieBLAS] No suitable " << Api::name
                << " algorithm found for " << "transA=" << transA
                << " transB=" << transB << " epilogue=" << epilogue << " A=["
                << shapeA.first << "x" << shapeA.second << "]" << " B=["
                << shapeB.first << "x" << shapeB.second << "]\n";
      exit(EXIT_FAILURE);
    }
    auto ins = algoCache.emplace(key, CacheEntry{h, {}}).first;
    if (algoCacheLimit) {
      lruOrder.push_front(key);
      ins->second.lru = lruOrder.begin();
      while (algoCache.size() > algoCacheLimit) {
        algoCache.erase(lruOrder.back());
        lruOrder.pop_back();
      }
    }
    return &ins->second.h;
  }

  void executeMatmul(typename Api::Operation transA,
                     typename Api::Operation transB,
                     typename Api::Epilogue epilogue, const void *alpha,
                     const void *A, const void *B, const void *beta,
                     const void *D_in, void *C_out, const void *bias_ptr,
                     const std::pair<std::size_t, std::size_t> &shapeA,
                     const std::pair<std::size_t, std::size_t> &shapeB,
                     const std::pair<std::size_t, std::size_t> &shapeC,
                     std::size_t ldA = 0, std::size_t ldB = 0,
                     decltype(Api::ComputeF32) computeType = Api::ComputeF32,
                     decltype(Api::RealF32) scaleType = Api::RealF32,
                     decltype(Api::RealF32) typeAB = Api::RealF32,
                     decltype(Api::RealF32) typeD = Api::RealF32,
                     decltype(Api::RealF32) biasType = Api::RealF32,
                     const float *scaleA = nullptr,
                     const float *scaleB = nullptr,
                     const float *scaleD = nullptr, bool alphaVector = false) {
    // Retrieve (or lazily compute) the cached algorithm for this shape
    auto *h = getOrComputeAlgo(transA, transB, epilogue, shapeA, shapeB, shapeC,
                               ldA, ldB, computeType, scaleType, typeAB, typeD,
                               biasType, alphaVector);

    // Retrieve the cached descriptor and patch the real bias pointer in-place
    auto &desc =
        getOrCreateDesc(transA, transB, epilogue, computeType, scaleType,
                        biasType, typeAB, typeD, alphaVector);
    if (bias_ptr && usesBias(epilogue)) {
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasPointer,
                                               &bias_ptr, sizeof(bias_ptr)));
    }
    // FP8: the scale pointers, set on every call like the bias pointer
    if (scaleA)
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescAScalePointer,
                                               &scaleA, sizeof(scaleA)));
    if (scaleB)
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBScalePointer,
                                               &scaleB, sizeof(scaleB)));
    if (scaleD)
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescDScalePointer,
                                               &scaleD, sizeof(scaleD)));

    auto lA = getOrCreateLayout(shapeA, ldA ? ldA : shapeA.first, typeAB);
    auto lB = getOrCreateLayout(shapeB, ldB ? ldB : shapeB.first, typeAB);
    auto lC = getOrCreateLayout(shapeC, shapeC.first, cTypeFor(typeD));
    auto lD = cTypeFor(typeD) == typeD
                  ? lC
                  : getOrCreateLayout(shapeC, shapeC.first, typeD);
    // C and D at one address need one layout; a C of another type than D
    // (FP8 D, beta 0) is never read, so it is left out
    if (lD != lC)
      D_in = nullptr;
    SOFIEBLAS_CHECK_LT(Api::matmul(ltHandle, desc, alpha, A, lA, B, lB, beta,
                                   D_in, lC, C_out, lD, &h->algo, d_workspace,
                                   workspaceSize, stream));
  }

  // the shared body of the fp8 GEMMs: E4M3 inputs, float sum and scale,
  // beta 0, BF16 bias, an output of type typeD (scaleD only for an FP8 one)
  void executeFp8(char transa, char transb, unsigned int m, unsigned int n,
                  unsigned int k, float alpha, const void *A,
                  const float *scaleA, const void *B, const float *scaleB,
                  const void *bias, void *C, decltype(Api::RealF32) typeD,
                  const float *scaleD, Epilogue epilogue) {
    const float beta = 0.f;
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  toApiEpilogue(epilogue), &alpha, A, B, &beta, C, C, bias,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n}, 0,
                  0, Api::ComputeF32, Api::RealF32, Api::RealF8E4M3, typeD,
                  Api::RealBF16, scaleA, scaleB, scaleD);
  }

  // the shared body of the quantized strided-batched multiplies; the bias is
  // shared by the batches and the scales of A and B are set for fp8
  void executeQuantBatched(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      const void *alpha, const void *A, long long strideA, const void *B,
      long long strideB, const void *beta, const void *bias, void *C,
      long long strideC, int batchCount, Epilogue epilogue,
      decltype(Api::ComputeF32) computeType, decltype(Api::RealF32) scaleType,
      decltype(Api::RealF32) typeAB, decltype(Api::RealF32) typeD,
      decltype(Api::RealF32) biasType, const float *scaleA = nullptr,
      const float *scaleB = nullptr) {
    auto *e = getOrComputeQuantBatched(
        transa, transb, m, n, k, strideA, strideB, strideC, batchCount,
        epilogue, computeType, scaleType, typeAB, typeD, biasType);
    const auto epi = toApiEpilogue(epilogue);
    auto &desc =
        getOrCreateDesc(charToTranspose(transa), charToTranspose(transb), epi,
                        computeType, scaleType, biasType, typeAB, typeD);
    if (bias && usesBias(epi)) {
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasPointer,
                                               &bias, sizeof(bias)));
      if constexpr (requires { Api::DescBiasBatchStride; }) {
        std::int64_t biasStride = 0;
        SOFIEBLAS_CHECK_LT(Api::descSetAttribute(
            desc, Api::DescBiasBatchStride, &biasStride, sizeof(biasStride)));
      }
    }
    if (scaleA)
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescAScalePointer,
                                               &scaleA, sizeof(scaleA)));
    if (scaleB)
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBScalePointer,
                                               &scaleB, sizeof(scaleB)));
    SOFIEBLAS_CHECK_LT(Api::matmul(ltHandle, desc, alpha, A, e->lA, B, e->lB,
                                   beta, C, e->lC, C, e->lD, &e->h.algo,
                                   d_workspace, workspaceSize, stream));
  }
};
