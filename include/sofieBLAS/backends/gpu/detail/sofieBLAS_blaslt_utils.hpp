#pragma once

// Helpers behind the GPU backends' public API
// the cache keys, and BlasLtBase, the part of the cuBLASLt / hipBLASLt wrapper
// that owns the handles and caches, builds layouts and descriptors, resolves
// algorithms and runs the multiplies. The Api table and the check macros
// SOFIEBLAS_CHECK_LT / SOFIEBLAS_CHECK_RT come from the vendor header, which
// includes this file after defining them.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sofieBLAS/backends/native/sofieBLAS_fallback.hpp"
#include "sofieBLAS/core.hpp"
#include "sofieBLAS/utils.hpp"
#include <alpaka/alpaka.hpp>

struct PairHash {
  std::size_t
  operator()(const std::pair<std::size_t, std::size_t> &p) const noexcept {
    std::size_t h1 = std::hash<std::size_t>{}(p.first);
    std::size_t h2 = std::hash<std::size_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

// one layout of a (rows, cols) matrix of an element type, with leading
// dimension ld (ld = rows for a dense one)
struct LayoutKey {
  int type;
  std::size_t rows, cols, ld;
  bool operator==(const LayoutKey &o) const noexcept {
    return type == o.type && rows == o.rows && cols == o.cols && ld == o.ld;
  }
};

struct LayoutKeyHash {
  std::size_t operator()(const LayoutKey &k) const noexcept {
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
  int typeInputs;      // backend data-type enum of the inputs A and B
  int typeD;           // backend data-type enum of the output D
  int alphaVector = 0; // 1 when alpha is a device vector, one value per row
  bool operator==(const DescKey &o) const noexcept {
    return transA == o.transA && transB == o.transB && epilogue == o.epilogue &&
           typeInputs == o.typeInputs && typeD == o.typeD &&
           alphaVector == o.alphaVector;
  }
};

struct DescKeyHash {
  std::size_t operator()(const DescKey &k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.transA) * 97u +
                    static_cast<std::size_t>(k.transB) * 31u +
                    static_cast<std::size_t>(k.epilogue);
    h = h * 31u + static_cast<std::size_t>(k.typeInputs);
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

enum class GemmMode { Vendor, Vendor_withExtendedEpilogue, Native };

struct GemmModeByEpilogue {
  std::array<GemmMode, 6> modes;
  GemmModeByEpilogue() { modes.fill(GemmMode::Native); }
  GemmMode &operator[](Epilogue e) {
    return modes[static_cast<std::size_t>(e)];
  }
  GemmMode operator[](Epilogue e) const {
    return modes[static_cast<std::size_t>(e)];
  }
  void fill(GemmMode mode) { modes.fill(mode); }
};

struct BlasLtCapabilities {
  GemmMode int8Int32 = GemmMode::Native;
  GemmMode fp16Fp32 = GemmMode::Native;
  GemmModeByEpilogue int8Int8;
  GemmModeByEpilogue int8Int8VectorAlpha;
  GemmModeByEpilogue fp8Fp32;
  GemmModeByEpilogue fp8Fp16;
  GemmModeByEpilogue fp8Bf16;
  GemmModeByEpilogue fp8Fp8;
  GemmModeByEpilogue fp8E5m2;
};

template <class Api> class BlasLtBase {
protected:
  using Fallback = FallbackBlas<typename Api::Tag, typename Api::Queue>;
  using Native = typename Fallback::Native;
  Fallback fallback;
  BlasLtCapabilities caps;
  bool quantization = false;

  typename Api::Handle ltHandle = nullptr;
  typename Api::BlasHandle handle = nullptr;
  typename Api::Preference preference = nullptr;
  void *d_workspace = nullptr;
  size_t workspaceSize = 1u << 25; // 32 MB
  typename Api::Stream stream = nullptr;

  // every layout the GEMMs use, dense or with a padded leading dimension
  std::unordered_map<LayoutKey, typename Api::Layout, LayoutKeyHash>
      layoutStore;

  // One cache entry per exact GEMM configuration: the heuristic result to
  // reuse, plus this entry's position in the recency list so a hit can mark
  // itself most-recently-used in O(1). The position is only maintained when a
  // cache limit is set; with no limit the list stays empty.
  //
  // The entry also holds the descriptor and the four layouts of its
  // configuration, so a call finds everything with the one cache lookup. They
  // live in descStore and layoutStore until the destructor, so the pointers
  // stay valid when the entry is evicted.
  struct DescEntry;
  struct CacheEntry {
    typename Api::HeuristicResult h{};
    std::list<AlgoKey>::iterator lru{};
    bool found = true;
    DescEntry *desc = nullptr;
    typename Api::Layout lA = nullptr, lB = nullptr, lC = nullptr, lD = nullptr;
  };

  // A descriptor and the pointers last set on it. Every set is a call into
  // the library, and a loop that multiplies with the same bias and scales
  // sets the same pointers on every call, so setting the same pointer again
  // is skipped (see patchPointer).
  struct DescEntry {
    typename Api::MatmulDesc desc = nullptr;
    const void *bias = unsetPointer();
    const void *scaleA = unsetPointer();
    const void *scaleB = unsetPointer();
    const void *scaleD = unsetPointer();
    std::int64_t biasStride = -1; // -1: not set yet
    // never equal to a pointer a caller passes
    static const void *unsetPointer() {
      static const char unset{};
      return &unset;
    }
  };

  // The descriptors and the algorithms of every GEMM configuration; the input
  // and output types are part of the keys, and the output type decides the
  // other types of a configuration. Layouts are keyed by their element type.
  std::unordered_map<DescKey, DescEntry, DescKeyHash> descStore;
  std::unordered_map<AlgoKey, CacheEntry, AlgoKeyHash> algoCache;
  // entries ordered most- to least-recently used; drives eviction
  std::list<AlgoKey> lruOrder;
  // 0 = unbounded; the limit applies to the algorithm cache as a whole
  std::size_t algoCacheLimit = 0;

public:
  BlasLtBase(const BlasLtBase &) = delete;
  BlasLtBase &operator=(const BlasLtBase &) = delete;
  BlasLtBase(BlasLtBase &&) = delete;
  BlasLtBase &operator=(BlasLtBase &&) = delete;

  BlasLtBase(typename Api::Queue &queue, std::size_t cacheLimit = 0)
      : fallback{queue}, algoCacheLimit{cacheLimit}, m_queue{queue} {
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

  const BlasLtCapabilities &capabilities() const { return caps; }

  bool quantizationEnabled() const { return quantization; }

  ~BlasLtBase() {
    for (auto &[key, layout] : layoutStore)
      if (layout)
        Api::layoutDestroy(layout);
    for (auto &[key, entry] : batchedStore) {
      Api::layoutDestroy(entry.lA);
      Api::layoutDestroy(entry.lB);
      Api::layoutDestroy(entry.lC);
    }
    for (auto &[key, entry] : quantBatchedStore)
      destroyQuantBatched(entry);
    for (auto &[key, entry] : descStore)
      if (entry.desc)
        Api::descDestroy(entry.desc);
    if (preference)
      Api::prefDestroy(preference);
    if (ltHandle)
      Api::ltDestroy(ltHandle);
    if (handle)
      Api::blasDestroy(handle);
    // a destructor can neither throw nor recover: report a failed free and go on
    if (d_workspace) {
      const auto status = Api::rtFree(d_workspace);
      if (status != Api::RtSuccess)
        std::cerr << "sofieBLAS: freeing the workspace threw errors: "
                  << Api::rtErrorString(status) << "\n";
    }
  }

protected:
  typename Api::Queue m_queue;

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
    case Epilogue::Gelu:
      return Api::EpilogueGelu;
    case Epilogue::Default:
      break;
    }
    return Api::EpilogueDefault;
  }

  // true for the epilogues that add a bias vector
  static bool usesBias(typename Api::Epilogue epilogue) {
    return epilogue != Api::EpilogueDefault && epilogue != Api::EpilogueRelu &&
           epilogue != Api::EpilogueGelu;
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
    bool found = true;
    DescEntry *desc = nullptr;
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
      decltype(Api::RealF32) typeD, decltype(Api::RealF32) biasType,
      bool required = true) {
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
    if (it != quantBatchedStore.end()) {
      if (it->second.found)
        return &it->second;
      if (!required)
        return nullptr;
      utils::reportQuantBatchedMissing<Api>(m, n, k, batchCount, strideA,
                                            strideB);
    }

    const auto shapeA = layoutKeyA(transa, m, k);
    const auto shapeB = layoutKeyB(transb, k, n);
    const std::pair<std::size_t, std::size_t> shapeC{m, n};
    auto &desc = getOrCreateDesc(opA, opB, epi, computeType, scaleType,
                                 biasType, typeAB, typeD);
    QuantBatchedEntry e{};
    e.desc = &desc;
    e.lA =
        createBatchedLayout(shapeA, shapeA.first, strideA, batchCount, typeAB);
    e.lB =
        createBatchedLayout(shapeB, shapeB.first, strideB, batchCount, typeAB);
    e.lC = createBatchedLayout(shapeC, m, strideC, batchCount, cTypeFor(typeD));
    e.lD = cTypeFor(typeD) == typeD
               ? e.lC
               : createBatchedLayout(shapeC, m, strideC, batchCount, typeD);
    int returned = 0;
    const auto status = Api::getHeuristic(ltHandle, desc.desc, e.lA, e.lB, e.lC,
                                          e.lD, preference, 1, &e.h, &returned);
    SOFIEBLAS_CHECK_LT(status);
    e.found = returned != 0;
    auto *entry = &quantBatchedStore.emplace(key, e).first->second;
    if (e.found)
      return entry;
    if (!required)
      return nullptr;
    utils::reportQuantBatchedMissing<Api>(m, n, k, batchCount, strideA,
                                          strideB);
  }

  // Returns the layout describing a (rows, cols) matrix of the given element
  // type with leading dimension ld (column-major), creating and caching it on
  // first use.
  typename Api::Layout
  getOrCreateLayout(const std::pair<std::size_t, std::size_t> &shape,
                    std::size_t ld,
                    decltype(Api::RealF32) type = Api::RealF32) {
    if (ld < shape.first)
      throw std::invalid_argument(
          std::string("sofieBLAS: leading dimension ") + std::to_string(ld) +
          " is smaller than the number of rows " + std::to_string(shape.first) +
          " of the matrix.");
    const LayoutKey key{static_cast<int>(type), shape.first, shape.second, ld};
    auto it = layoutStore.find(key);
    if (it != layoutStore.end())
      return it->second;
    typename Api::Layout layout = nullptr;
    SOFIEBLAS_CHECK_LT(
        Api::layoutCreate(&layout, type, shape.first, shape.second, ld));
    layoutStore.emplace(key, layout);
    return layout;
  }

  // Sets a pointer attribute of a descriptor, unless that pointer is already
  // set: the entry remembers the last one (field is the member that holds it)
  template <typename TAttr>
  void patchPointer(DescEntry &entry, TAttr attribute,
                    const void *DescEntry::*field, const void *pointer) {
    if (entry.*field == pointer)
      return;
    SOFIEBLAS_CHECK_LT(Api::descSetAttribute(entry.desc, attribute, &pointer,
                                             sizeof(pointer)));
    entry.*field = pointer;
  }

  // Sets the stride between the biases of the batches, where the library has
  // that attribute (a bias shared by all batches has stride 0)
  void patchBiasStride([[maybe_unused]] DescEntry &entry,
                       [[maybe_unused]] std::int64_t stride) {
    if constexpr (requires { Api::DescBiasBatchStride; }) {
      if (entry.biasStride == stride)
        return;
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(
          entry.desc, Api::DescBiasBatchStride, &stride, sizeof(stride)));
      entry.biasStride = stride;
    }
  }

  // the pointers to the scales of A, B and D (a null one is left unset)
  void patchScales(DescEntry &entry, const float *scaleA, const float *scaleB,
                   const float *scaleD) {
    if (scaleA)
      patchPointer(entry, Api::DescAScalePointer, &DescEntry::scaleA, scaleA);
    if (scaleB)
      patchPointer(entry, Api::DescBScalePointer, &DescEntry::scaleB, scaleB);
    if (scaleD)
      patchPointer(entry, Api::DescDScalePointer, &DescEntry::scaleD, scaleD);
  }

  DescEntry &getOrCreateDesc(
      typename Api::Operation transA, typename Api::Operation transB,
      typename Api::Epilogue epilogue,
      decltype(Api::ComputeF32) computeType = Api::ComputeF32,
      decltype(Api::RealF32) scaleType = Api::RealF32,
      decltype(Api::RealF32) biasType = Api::RealF32,
      decltype(Api::RealF32) typeAB = Api::RealF32,
      decltype(Api::RealF32) typeD = Api::RealF32, bool alphaVector = false) {
    DescKey key{(int)transA, (int)transB, (int)epilogue,
                (int)typeAB, (int)typeD,  alphaVector ? 1 : 0};
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
    DescEntry entry;
    entry.desc = desc;
    return descStore.emplace(key, entry).first->second;
  }

  CacheEntry *
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
                   bool alphaVector = false, bool required = true) {
    // ldA/ldB are 0 for dense operands
    if (ldA == shapeA.first)
      ldA = 0;
    if (ldB == shapeB.first)
      ldB = 0;
    AlgoKey key{{(int)transA, (int)transB, (int)epilogue, (int)typeAB,
                 (int)typeD, alphaVector ? 1 : 0},
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
      if (it->second.found)
        return &it->second;
      if (!required)
        return nullptr;
      utils::reportAlgoMissing<Api>(transA, transB, epilogue, shapeA, shapeB);
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
    const auto status = Api::getHeuristic(ltHandle, desc.desc, lA, lB, lC, lD,
                                          preference, 1, &h, &returnedResults);
    SOFIEBLAS_CHECK_LT(status);
    if (returnedResults == 0 && required)
      utils::reportAlgoMissing<Api>(transA, transB, epilogue, shapeA, shapeB);
    CacheEntry entry{h, {}, returnedResults != 0};
    entry.desc = &desc;
    entry.lA = lA;
    entry.lB = lB;
    entry.lC = lC;
    entry.lD = lD;
    auto ins = algoCache.emplace(key, entry).first;
    if (algoCacheLimit) {
      lruOrder.push_front(key);
      ins->second.lru = lruOrder.begin();
      while (algoCache.size() > algoCacheLimit) {
        algoCache.erase(lruOrder.back());
        lruOrder.pop_back();
      }
    }
    return returnedResults != 0 ? &ins->second : nullptr;
  }

  // the cached descriptor, algorithm and layouts of a multiply
  struct MatmulSetup {
    DescEntry *desc = nullptr;
    typename Api::HeuristicResult *algo = nullptr;
    typename Api::Layout lA = nullptr, lB = nullptr, lC = nullptr, lD = nullptr;
  };

  // Looks up (or computes) the algorithm and the descriptor of a multiply and
  // patches the real bias pointer into the cached descriptor. The algorithm is
  // null when the library has no kernel for the configuration.
  MatmulSetup
  prepareMatmul(typename Api::Operation transA, typename Api::Operation transB,
                typename Api::Epilogue epilogue, const void *bias_ptr,
                const std::pair<std::size_t, std::size_t> &shapeA,
                const std::pair<std::size_t, std::size_t> &shapeB,
                const std::pair<std::size_t, std::size_t> &shapeC,
                std::size_t ldA, std::size_t ldB,
                decltype(Api::ComputeF32) computeType,
                decltype(Api::RealF32) scaleType, decltype(Api::RealF32) typeAB,
                decltype(Api::RealF32) typeD, decltype(Api::RealF32) biasType,
                bool alphaVector, bool required) {
    auto *entry = getOrComputeAlgo(
        transA, transB, epilogue, shapeA, shapeB, shapeC, ldA, ldB, computeType,
        scaleType, typeAB, typeD, biasType, alphaVector, required);
    if (!entry)
      return {};
    if (bias_ptr && usesBias(epilogue))
      patchPointer(*entry->desc, Api::DescBiasPointer, &DescEntry::bias,
                   bias_ptr);
    return {entry->desc, &entry->h, entry->lA, entry->lB, entry->lC, entry->lD};
  }

  // the matmul call of a prepared multiply
  bool launchMatmul(const MatmulSetup &setup, const void *alpha, const void *A,
                    const void *B, const void *beta, const void *D_in,
                    void *C_out, bool required) {
    // C and D at one address need one layout; a C of another type than D
    // (FP8 D, beta 0) is never read, so it is left out
    if (setup.lD != setup.lC)
      D_in = nullptr;
    const auto status =
        Api::matmul(ltHandle, setup.desc->desc, alpha, A, setup.lA, B, setup.lB,
                    beta, D_in, setup.lC, C_out, setup.lD, &setup.algo->algo,
                    d_workspace, workspaceSize, stream);
    if (required)
      SOFIEBLAS_CHECK_LT(status);
    return status == Api::StatusSuccess;
  }

  bool executeMatmul(typename Api::Operation transA,
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
                     bool alphaVector = false, bool required = true) {
    const auto setup = prepareMatmul(
        transA, transB, epilogue, bias_ptr, shapeA, shapeB, shapeC, ldA, ldB,
        computeType, scaleType, typeAB, typeD, biasType, alphaVector, required);
    if (!setup.algo)
      return false;
    return launchMatmul(setup, alpha, A, B, beta, D_in, C_out, required);
  }

  // executeMatmul for the quantized multiplies that scale: the device pointers
  // to the scales of A, B and D (a null one is left unset) are patched into
  // the cached descriptor on every call
  bool executeMatmulWithScale(
      typename Api::Operation transA, typename Api::Operation transB,
      typename Api::Epilogue epilogue, const void *alpha, const void *A,
      const void *B, const void *beta, const void *D_in, void *C_out,
      const void *bias_ptr, const std::pair<std::size_t, std::size_t> &shapeA,
      const std::pair<std::size_t, std::size_t> &shapeB,
      const std::pair<std::size_t, std::size_t> &shapeC, std::size_t ldA,
      std::size_t ldB, decltype(Api::ComputeF32) computeType,
      decltype(Api::RealF32) scaleType, decltype(Api::RealF32) typeAB,
      decltype(Api::RealF32) typeD, decltype(Api::RealF32) biasType,
      const float *scaleA, const float *scaleB, const float *scaleD,
      bool alphaVector = false, bool required = true) {
    const auto setup = prepareMatmul(
        transA, transB, epilogue, bias_ptr, shapeA, shapeB, shapeC, ldA, ldB,
        computeType, scaleType, typeAB, typeD, biasType, alphaVector, required);
    if (!setup.algo)
      return false;
    patchScales(*setup.desc, scaleA, scaleB, scaleD);
    return launchMatmul(setup, alpha, A, B, beta, D_in, C_out, required);
  }

  static GemmMode extendedEpilogueMode(GemmMode plainGemm) {
    return plainGemm == GemmMode::Vendor ? GemmMode::Vendor_withExtendedEpilogue
                                         : GemmMode::Native;
  }

  const GemmModeByEpilogue &fp8Modes(decltype(Api::RealF32) typeD) const {
    if (typeD == Api::RealF16)
      return caps.fp8Fp16;
    if (typeD == Api::RealBF16)
      return caps.fp8Bf16;
    if (typeD == Api::RealF8E4M3)
      return caps.fp8Fp8;
    if (typeD == Api::RealF8E5M2)
      return caps.fp8E5m2;
    return caps.fp8Fp32;
  }

  template <typename TIn, typename TSum>
  bool vendorSum(char transa, char transb, unsigned int m, unsigned int n,
                 unsigned int k, const TIn *A, long long strideA, const TIn *B,
                 long long strideB, TSum *C, long long strideC,
                 int batchCount) {
    constexpr bool integer = std::is_same_v<TSum, std::int32_t>;
    const auto compute = integer ? Api::ComputeI32 : Api::ComputeF32;
    const auto typeAB = integer ? Api::RealI8 : Api::RealF16;
    const auto typeSum = integer ? Api::RealI32 : Api::RealF32;
    const TSum one = 1, zero = 0;
    if (auto *batched =
            batchCount > 1
                ? getOrComputeQuantBatched(transa, transb, m, n, k, strideA,
                                           strideB, strideC, batchCount,
                                           Epilogue::Default, compute, typeSum,
                                           typeAB, typeSum, Api::RealF32, false)
                : nullptr) {
      executeQuantBatched(*batched, &one, A, B, &zero, nullptr, C,
                          Epilogue::Default);
      return true;
    }
    for (int b = 0; b < batchCount; ++b)
      if (!executeMatmul(charToTranspose(transa), charToTranspose(transb),
                         Api::EpilogueDefault, &one, A + b * strideA,
                         B + b * strideB, &zero, C + b * strideC,
                         C + b * strideC, nullptr, layoutKeyA(transa, m, k),
                         layoutKeyB(transb, k, n), {m, n}, 0, 0, compute,
                         typeSum, typeAB, typeSum, Api::RealF32, false, false))
        return false;
    return true;
  }

  struct SafeVendor {
    using Half = __half;
    BlasLtBase &base;

    bool int8Int32Available() const {
      return base.caps.int8Int32 == GemmMode::Vendor;
    }
    bool fp16Fp32Available() const {
      return base.caps.fp16Fp32 == GemmMode::Vendor;
    }
    bool int8Int32(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, const std::int8_t *A, long long strideA,
                   const std::int8_t *B, long long strideB, std::int32_t *C,
                   long long strideC, int batchCount) {
      return int8Int32Available() &&
             base.vendorSum(transa, transb, m, n, k, A, strideA, B, strideB, C,
                            strideC, batchCount);
    }
    bool fp16Fp32(char transa, char transb, unsigned int m, unsigned int n,
                  unsigned int k, const Half *A, long long strideA,
                  const Half *B, long long strideB, float *C, long long strideC,
                  int batchCount) {
      return fp16Fp32Available() &&
             base.vendorSum(transa, transb, m, n, k, A, strideA, B, strideB, C,
                            strideC, batchCount);
    }
  };

  SafeVendor safeVendor() { return SafeVendor{*this}; }

  // the shared body of the fp8 GEMMs: E4M3 inputs, float sum and scale,
  // beta 0, BF16 bias, an output of type typeD (scaleD only for an FP8 one)
  template <typename TOut>
  void executeFp8(char transa, char transb, unsigned int m, unsigned int n,
                  unsigned int k, float alpha, const typename Api::Fp8E4M3 *A,
                  const float *scaleA, const typename Api::Fp8E4M3 *B,
                  const float *scaleB, const typename Api::Bf16 *bias, TOut *C,
                  decltype(Api::RealF32) typeD, const float *scaleD,
                  Epilogue epilogue) {
    const auto opA = charToTranspose(transa), opB = charToTranspose(transb);
    const auto epi = toApiEpilogue(epilogue);
    const auto shapeA = layoutKeyA(transa, m, k);
    const auto shapeB = layoutKeyB(transb, k, n);
    if (fp8Modes(typeD)[epilogue] == GemmMode::Vendor) {
      const auto setup = prepareMatmul(
          opA, opB, epi, bias, shapeA, shapeB, {m, n}, 0, 0, Api::ComputeF32,
          Api::RealF32, Api::RealF8E4M3, typeD, Api::RealBF16, false, false);
      if (setup.algo) {
        const float beta = 0.f;
        patchScales(*setup.desc, scaleA, scaleB, scaleD);
        launchMatmul(setup, &alpha, A, B, &beta, C, C, true);
        return;
      }
    }
    fallback.fp8(safeVendor(), transa, transb, m, n, k, A, 0, B, 0, C,
                 static_cast<long long>(m) * n, 1,
                 Fallback::epilogueParams(epilogue, alpha, nullptr, scaleA,
                                          scaleB, scaleD, bias));
  }

  void executeFp8Batched(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k, float alpha,
                         const typename Api::Fp8E4M3 *A, const float *scaleA,
                         long long strideA, const typename Api::Fp8E4M3 *B,
                         const float *scaleB, long long strideB,
                         const typename Api::Bf16 *bias, float *C,
                         long long strideC, int batchCount, Epilogue epilogue) {
    if (auto *batched =
            caps.fp8Fp32[epilogue] == GemmMode::Vendor
                ? getOrComputeQuantBatched(
                      transa, transb, m, n, k, strideA, strideB, strideC,
                      batchCount, epilogue, Api::ComputeF32, Api::RealF32,
                      Api::RealF8E4M3, Api::RealF32, Api::RealBF16, false)
                : nullptr) {
      const float beta = 0.f;
      executeQuantBatched(*batched, &alpha, A, B, &beta, bias, C, epilogue,
                          scaleA, scaleB);
      return;
    }
    fallback.fp8(safeVendor(), transa, transb, m, n, k, A, strideA, B, strideB,
                 C, strideC, batchCount,
                 Fallback::epilogueParams(epilogue, alpha, nullptr, scaleA,
                                          scaleB, nullptr, bias));
  }

  void executeInt8Int32(char transa, char transb, unsigned int m,
                        unsigned int n, unsigned int k, const std::int8_t *A,
                        long long strideA, const std::int8_t *B,
                        long long strideB, std::int32_t *C, long long strideC,
                        int batchCount) {
    fallback.int8Int32(
        safeVendor(), transa, transb, m, n, k, A, strideA, B, strideB, C,
        batchCount == 1 ? static_cast<long long>(m) * n : strideC, batchCount);
  }

  void executeInt8Int8(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, const float *alphaVector,
                       const std::int8_t *A, const std::int8_t *B,
                       const float *bias, std::int8_t *C, Epilogue epilogue) {
    const auto opA = charToTranspose(transa), opB = charToTranspose(transb);
    const auto epi = toApiEpilogue(epilogue);
    const auto shapeA = layoutKeyA(transa, m, k);
    const auto shapeB = layoutKeyB(transb, k, n);
    const bool vector = alphaVector != nullptr;
    if ((vector ? caps.int8Int8VectorAlpha : caps.int8Int8)[epilogue] ==
        GemmMode::Vendor) {
      const auto setup = prepareMatmul(
          opA, opB, epi, bias, shapeA, shapeB, {m, n}, 0, 0, Api::ComputeI32,
          Api::RealF32, Api::RealI8, Api::RealI8, Api::RealF32, vector, false);
      if (setup.algo) {
        const float beta = 0.f;
        launchMatmul(setup,
                     vector ? static_cast<const void *>(alphaVector) : &alpha,
                     A, B, &beta, C, C, true);
        return;
      }
    }
    fallback.int8Int8(safeVendor(), transa, transb, m, n, k, A, 0, B, 0, C,
                      static_cast<long long>(m) * n, 1,
                      Fallback::epilogueParams(epilogue, alpha, alphaVector,
                                               nullptr, nullptr, nullptr,
                                               bias));
  }

  void executeInt8Int8Batched(char transa, char transb, unsigned int m,
                              unsigned int n, unsigned int k, float alpha,
                              const std::int8_t *A, long long strideA,
                              const std::int8_t *B, long long strideB,
                              const float *bias, std::int8_t *C,
                              long long strideC, int batchCount,
                              Epilogue epilogue) {
    if (auto *batched =
            caps.int8Int8[epilogue] == GemmMode::Vendor
                ? getOrComputeQuantBatched(
                      transa, transb, m, n, k, strideA, strideB, strideC,
                      batchCount, epilogue, Api::ComputeI32, Api::RealF32,
                      Api::RealI8, Api::RealI8, Api::RealF32, false)
                : nullptr) {
      const float beta = 0.f;
      executeQuantBatched(*batched, &alpha, A, B, &beta, bias, C, epilogue);
      return;
    }
    fallback.int8Int8(safeVendor(), transa, transb, m, n, k, A, strideA, B,
                      strideB, C, strideC, batchCount,
                      Fallback::epilogueParams(epilogue, alpha, nullptr,
                                               nullptr, nullptr, nullptr,
                                               bias));
  }

  static float applyEpilogue(float x, float bias, Epilogue epilogue) {
    if (epilogue == Epilogue::Bias || epilogue == Epilogue::ReluBias ||
        epilogue == Epilogue::GeluBias)
      x += bias;
    if (epilogue == Epilogue::Relu || epilogue == Epilogue::ReluBias)
      x = x > 0.f ? x : 0.f;
    else if (epilogue == Epilogue::GeluBias || epilogue == Epilogue::Gelu)
      x *= 0.5f *
           (1.f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    return x;
  }

  void requireQuantization() const {
    if (!quantization)
      throw std::logic_error("sofieBLAS: the quantized methods need a backend "
                             "constructed with {.quantization = true}");
  }

  void enableQuantization(bool forceNative) {
    quantization = true;
    if (forceNative)
      return;
    if constexpr (Api::ProbeCapabilities) {
      detectCapabilities();
    } else {
      caps.int8Int32 = GemmMode::Vendor;
      caps.fp16Fp32 = GemmMode::Vendor;
      for (auto *table :
           {&caps.int8Int8, &caps.int8Int8VectorAlpha, &caps.fp8Fp32,
            &caps.fp8Fp16, &caps.fp8Bf16, &caps.fp8Fp8, &caps.fp8E5m2})
        table->fill(GemmMode::Vendor);
      if constexpr (!(requires { Api::PointerModeAlphaDeviceVectorBetaHost; }))
        caps.int8Int8VectorAlpha.fill(GemmMode::Vendor_withExtendedEpilogue);
    }
  }

  void detectCapabilities() {
    constexpr std::size_t m = 32, n = 32, k = 32;
    const std::pair<std::size_t, std::size_t> shapeA{k, m}, shapeB{k, n},
        shapeC{m, n};
    constexpr Epilogue epilogues[] = {Epilogue::Default,  Epilogue::Bias,
                                      Epilogue::ReluBias, Epilogue::GeluBias,
                                      Epilogue::Relu,     Epilogue::Gelu};
    const auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto upload = [&](const auto &v) {
      using T = typename std::decay_t<decltype(v)>::value_type;
      auto d =
          alpaka::allocAsyncBufIfSupported<T, std::size_t>(m_queue, v.size());
      alpaka::memcpy(m_queue, d, alpaka::createView(host, v.data(), v.size()));
      alpaka::wait(m_queue);
      return d;
    };
    auto download = [&](auto &d, auto &v) {
      auto view = alpaka::createView(host, v.data(), v.size());
      alpaka::memcpy(m_queue, view, d);
      alpaka::wait(m_queue);
    };
    auto reset = [&](auto &d, const auto &v) {
      alpaka::memcpy(m_queue, d, alpaka::createView(host, v.data(), v.size()));
    };

    std::vector<int> a(k * m), b(k * n), sums(m * n);
    std::vector<float> alphas(m), bias(m);
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = static_cast<int>(i * 5 % 7) - 3;
    for (std::size_t i = 0; i < b.size(); ++i)
      b[i] = static_cast<int>(i * 3 % 5) - 2;
    for (std::size_t i = 0; i < m; ++i) {
      alphas[i] = 0.125f * static_cast<float>(i % 4 + 2);
      bias[i] = -4.f + 0.5f * static_cast<float>(i % 16);
    }
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t p = 0; p < k; ++p)
          sums[j * m + i] += a[i * k + p] * b[j * k + p];

    std::vector<std::int8_t> a8(a.begin(), a.end()), b8(b.begin(), b.end());
    const std::vector<std::int32_t> poison32(m * n, -77);
    std::vector<std::int32_t> actual32(m * n);
    auto dA = upload(a8);
    auto dB = upload(b8);
    auto dC32 = upload(poison32);
    {
      const std::int32_t one = 1, zero = 0;
      const bool ran = executeMatmul(
          Api::OpT, Api::OpN, Api::EpilogueDefault, &one,
          alpaka::getPtrNative(dA), alpaka::getPtrNative(dB), &zero,
          alpaka::getPtrNative(dC32), alpaka::getPtrNative(dC32), nullptr,
          shapeA, shapeB, shapeC, 0, 0, Api::ComputeI32, Api::RealI32,
          Api::RealI8, Api::RealI32, Api::RealF32, false, false);
      if (ran)
        download(dC32, actual32);
      caps.int8Int32 =
          ran && std::equal(actual32.begin(), actual32.end(), sums.begin())
              ? GemmMode::Vendor
              : GemmMode::Native;
    }

    std::vector<__half> ah(a.size()), bh(b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      ah[i] = static_cast<__half>(static_cast<float>(a[i]));
    for (std::size_t i = 0; i < b.size(); ++i)
      bh[i] = static_cast<__half>(static_cast<float>(b[i]));
    const std::vector<float> poisonF(m * n, -77.f);
    std::vector<float> actualF(m * n);
    auto dAh = upload(ah);
    auto dBh = upload(bh);
    auto dCF = upload(poisonF);
    {
      const float one = 1.f, zero = 0.f;
      const bool ran = executeMatmul(
          Api::OpT, Api::OpN, Api::EpilogueDefault, &one,
          alpaka::getPtrNative(dAh), alpaka::getPtrNative(dBh), &zero,
          alpaka::getPtrNative(dCF), alpaka::getPtrNative(dCF), nullptr, shapeA,
          shapeB, shapeC, 0, 0, Api::ComputeF32, Api::RealF32, Api::RealF16,
          Api::RealF32, Api::RealF32, false, false);
      bool ok = ran;
      if (ran) {
        download(dCF, actualF);
        for (std::size_t e = 0; e < m * n && ok; ++e)
          ok = std::abs(actualF[e] - static_cast<float>(sums[e])) <=
               1e-3f * (1.f + std::abs(static_cast<float>(sums[e])));
      }
      caps.fp16Fp32 = ok ? GemmMode::Vendor : GemmMode::Native;
    }

    const std::vector<std::int8_t> poison8(m * n, std::int8_t{-77});
    std::vector<std::int8_t> actual8(m * n);
    auto dAlpha = upload(alphas);
    auto dBias = upload(bias);
    auto dC = upload(poison8);
    for (bool vector : {false, true}) {
      auto &table = vector ? caps.int8Int8VectorAlpha : caps.int8Int8;
      table.fill(extendedEpilogueMode(caps.int8Int32));
      if constexpr (!(requires { Api::PointerModeAlphaDeviceVectorBetaHost; }))
        if (vector)
          continue;
      for (Epilogue epilogue : epilogues) {
        reset(dC, poison8);
        const float alpha = 0.5f, beta = 0.f;
        if (!executeMatmul(
                Api::OpT, Api::OpN, toApiEpilogue(epilogue),
                vector ? static_cast<const void *>(alpaka::getPtrNative(dAlpha))
                       : static_cast<const void *>(&alpha),
                alpaka::getPtrNative(dA), alpaka::getPtrNative(dB), &beta,
                alpaka::getPtrNative(dC), alpaka::getPtrNative(dC),
                alpaka::getPtrNative(dBias), shapeA, shapeB, shapeC, 0, 0,
                Api::ComputeI32, Api::RealF32, Api::RealI8, Api::RealI8,
                Api::RealF32, vector, false))
          continue;
        download(dC, actual8);
        const int tolerance =
            epilogue == Epilogue::GeluBias || epilogue == Epilogue::Gelu ? 1
                                                                         : 0;
        bool ok = true;
        for (std::size_t e = 0; e < m * n && ok; ++e) {
          const std::size_t i = e % m;
          const float x = std::nearbyint(applyEpilogue(
              (vector ? alphas[i] : alpha) * static_cast<float>(sums[e]),
              bias[i], epilogue));
          const int expected =
              static_cast<int>(std::min(127.f, std::max(-128.f, x)));
          ok = std::abs(static_cast<int>(actual8[e]) - expected) <= tolerance;
        }
        if (ok)
          table[epilogue] = GemmMode::Vendor;
      }
    }

    using Fp8 = typename Api::Fp8E4M3;
    using Bf16 = typename Api::Bf16;
    std::vector<Fp8> af(a.size()), bf(b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      af[i] = static_cast<Fp8>(static_cast<float>(a[i]));
    for (std::size_t i = 0; i < b.size(); ++i)
      bf[i] = static_cast<Fp8>(static_cast<float>(b[i]));
    std::vector<Bf16> biasBf(m);
    for (std::size_t i = 0; i < m; ++i)
      biasBf[i] = static_cast<Bf16>(bias[i]);
    const std::vector<float> scales{0.5f, 2.f};
    auto dAf = upload(af);
    auto dBf = upload(bf);
    auto dBiasBf = upload(biasBf);
    auto dScales = upload(scales);
    const float *scaleA = alpaka::getPtrNative(dScales);
    const float *scaleB = scaleA + 1;
    for (auto *table : {&caps.fp8Fp32, &caps.fp8Fp16, &caps.fp8Bf16,
                        &caps.fp8Fp8, &caps.fp8E5m2})
      table->fill(extendedEpilogueMode(caps.fp16Fp32));
    for (Epilogue epilogue : epilogues) {
      reset(dCF, poisonF);
      const float alpha = 0.25f, beta = 0.f;
      if (!executeMatmulWithScale(
              Api::OpT, Api::OpN, toApiEpilogue(epilogue), &alpha,
              alpaka::getPtrNative(dAf), alpaka::getPtrNative(dBf), &beta,
              alpaka::getPtrNative(dCF), alpaka::getPtrNative(dCF),
              alpaka::getPtrNative(dBiasBf), shapeA, shapeB, shapeC, 0, 0,
              Api::ComputeF32, Api::RealF32, Api::RealF8E4M3, Api::RealF32,
              Api::RealBF16, scaleA, scaleB, nullptr, false, false))
        continue;
      download(dCF, actualF);
      bool ok = true;
      for (std::size_t e = 0; e < m * n && ok; ++e) {
        const float expected = applyEpilogue(alpha * scales[0] * scales[1] *
                                             static_cast<float>(sums[e]),
                                         bias[e % m], epilogue);
        ok = std::abs(actualF[e] - expected) <= 1e-3f * (1.f + std::abs(expected));
      }
      if (!ok)
        continue;
      caps.fp8Fp32[epilogue] = GemmMode::Vendor;
      const auto apiEpilogue = toApiEpilogue(epilogue);
      for (auto [typeD, table] : {std::pair{Api::RealF16, &caps.fp8Fp16},
                                  std::pair{Api::RealBF16, &caps.fp8Bf16},
                                  std::pair{Api::RealF8E4M3, &caps.fp8Fp8},
                                  std::pair{Api::RealF8E5M2, &caps.fp8E5m2}})
        if (getOrComputeAlgo(Api::OpT, Api::OpN, apiEpilogue, shapeA, shapeB,
                             shapeC, 0, 0, Api::ComputeF32, Api::RealF32,
                             Api::RealF8E4M3, typeD, Api::RealBF16, false,
                             false))
          (*table)[epilogue] = GemmMode::Vendor;
    }
    algoCache.clear();
    lruOrder.clear();
  }

  // the shared body of the quantized strided-batched multiplies, on the cached
  // entry of the configuration (getOrComputeQuantBatched); the bias is shared
  // by the batches and the scales of A and B are set for fp8
  void executeQuantBatched(QuantBatchedEntry &e, const void *alpha,
                           const void *A, const void *B, const void *beta,
                           const void *bias, void *C, Epilogue epilogue,
                           const float *scaleA = nullptr,
                           const float *scaleB = nullptr) {
    auto &desc = *e.desc;
    if (bias && usesBias(toApiEpilogue(epilogue))) {
      patchPointer(desc, Api::DescBiasPointer, &DescEntry::bias, bias);
      patchBiasStride(desc, 0);
    }
    patchScales(desc, scaleA, scaleB, nullptr);
    SOFIEBLAS_CHECK_LT(Api::matmul(ltHandle, desc.desc, alpha, A, e.lA, B, e.lB,
                                   beta, C, e.lC, C, e.lD, &e.h.algo,
                                   d_workspace, workspaceSize, stream));
  }
};
