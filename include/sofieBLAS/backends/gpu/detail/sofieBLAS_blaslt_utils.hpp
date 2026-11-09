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
  operator()(const std::pair<std::size_t, std::size_t> &pair) const noexcept {
    std::size_t firstHash = std::hash<std::size_t>{}(pair.first);
    std::size_t secondHash = std::hash<std::size_t>{}(pair.second);
    return firstHash ^ (secondHash + 0x9e3779b97f4a7c15ULL + (firstHash << 6) +
                        (firstHash >> 2));
  }
};

// one layout of a (rows, cols) matrix of an element type, with leading
// dimension ld (ld = rows for a dense one)
struct LayoutKey {
  int elementType;
  std::size_t rows, cols, leadingDimension;
  bool operator==(const LayoutKey &other) const noexcept {
    return elementType == other.elementType && rows == other.rows &&
           cols == other.cols && leadingDimension == other.leadingDimension;
  }
};

struct LayoutKeyHash {
  std::size_t operator()(const LayoutKey &key) const noexcept {
    std::size_t hashVal =
        PairHash{}({key.rows, key.cols}) ^
        (std::hash<std::size_t>{}(key.leadingDimension) * 31u);
    return hashVal ^ (std::hash<int>{}(key.elementType) +
                      0x9e3779b97f4a7c15ULL + (hashVal << 6) + (hashVal >> 2));
  }
};

struct DescKey {
  int transA; // backend transpose enum encoded as int
  int transB;
  int epilogue;        // backend epilogue enum encoded as int
  int typeInputs;      // backend data-type enum of the inputs A and B
  int typeD;           // backend data-type enum of the output D
  int alphaVector = 0; // 1 when alpha is a device vector, one value per row
  bool operator==(const DescKey &other) const noexcept {
    return transA == other.transA && transB == other.transB &&
           epilogue == other.epilogue && typeInputs == other.typeInputs &&
           typeD == other.typeD && alphaVector == other.alphaVector;
  }
};

struct DescKeyHash {
  std::size_t operator()(const DescKey &key) const noexcept {
    std::size_t hashVal = static_cast<std::size_t>(key.transA) * 97u +
                          static_cast<std::size_t>(key.transB) * 31u +
                          static_cast<std::size_t>(key.epilogue);
    hashVal = hashVal * 31u + static_cast<std::size_t>(key.typeInputs);
    hashVal = hashVal * 31u + static_cast<std::size_t>(key.typeD);
    hashVal = hashVal * 31u + static_cast<std::size_t>(key.alphaVector);
    return hashVal ^ (hashVal >> 16);
  }
};

struct AlgoKey {
  DescKey descKey;
  std::size_t rowsA, colsA;     // physical dimensions of A in layoutStore
  std::size_t rowsB, colsB;     // physical dimensions of B in layoutStore
  std::size_t ldA = 0, ldB = 0; // leading dimensions of A and B, 0 when dense
  bool operator==(const AlgoKey &other) const noexcept {
    return descKey == other.descKey && rowsA == other.rowsA &&
           colsA == other.colsA && rowsB == other.rowsB &&
           colsB == other.colsB && ldA == other.ldA && ldB == other.ldB;
  }
};

struct AlgoKeyHash {
  std::size_t operator()(const AlgoKey &key) const noexcept {
    std::size_t hashVal = DescKeyHash{}(key.descKey);
    // folds the hash of one field into the running hash h, so that keys
    // differing in any field (or in field order) hash differently
    auto combineHash = [&](std::size_t fieldVal) {
      hashVal ^= std::hash<std::size_t>{}(fieldVal) + 0x9e3779b97f4a7c15ULL +
                 (hashVal << 6) + (hashVal >> 2);
    };
    combineHash(key.rowsA);
    combineHash(key.colsA);
    combineHash(key.rowsB);
    combineHash(key.colsB);
    combineHash(key.ldA);
    combineHash(key.ldB);
    return hashVal;
  }
};

enum class GemmMode { Vendor, Vendor_withExtendedEpilogue, Native };

struct GemmModeByEpilogue {
  std::array<GemmMode, 6> modes;
  GemmModeByEpilogue() { modes.fill(GemmMode::Native); }
  GemmMode &operator[](Epilogue epilogue) {
    return modes[static_cast<std::size_t>(epilogue)];
  }
  GemmMode operator[](Epilogue epilogue) const {
    return modes[static_cast<std::size_t>(epilogue)];
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
  BlasLtCapabilities detectedCapabilities;
  bool quantization = false;

  typename Api::Handle ltHandle = nullptr;
  typename Api::BlasHandle handle = nullptr;
  typename Api::Preference preference = nullptr;
  void *deviceWorkspace = nullptr;
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
    typename Api::HeuristicResult heuristicResult{};
    std::list<AlgoKey>::iterator lruPosition{};
    bool found = true;
    DescEntry *descEntry = nullptr;
    typename Api::Layout layoutA = nullptr, layoutB = nullptr,
                         layoutC = nullptr, layoutD = nullptr;
  };

  // A descriptor and the pointers last set on it. Every set is a call into
  // the library, and a loop that multiplies with the same bias and scales
  // sets the same pointers on every call, so setting the same pointer again
  // is skipped (see patchPtr).
  struct DescEntry {
    typename Api::MatmulDesc descriptor = nullptr;
    const void *bias = unsetPtr();
    const void *scaleA = unsetPtr();
    const void *scaleB = unsetPtr();
    const void *scaleD = unsetPtr();
    std::int64_t biasStride = -1; // -1: not set yet
    // never equal to a pointer a caller passes
    static const void *unsetPtr() {
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
    SOFIEBLAS_CHECK_RT(Api::rtMalloc(&deviceWorkspace, workspaceSize));
    SOFIEBLAS_CHECK_LT(Api::prefSetAttribute(preference, Api::PrefMaxWorkspace,
                                             &workspaceSize,
                                             sizeof(workspaceSize)));
  }

  const BlasLtCapabilities &capabilities() const {
    return detectedCapabilities;
  }

  bool quantizationEnabled() const { return quantization; }

  ~BlasLtBase() {
    for (auto &[key, layout] : layoutStore)
      if (layout)
        Api::layoutDestroy(layout);
    for (auto &[key, entry] : batchedStore) {
      Api::layoutDestroy(entry.layoutA);
      Api::layoutDestroy(entry.layoutB);
      Api::layoutDestroy(entry.layoutC);
    }
    for (auto &[key, entry] : quantBatchedStore)
      destroyQuantBatched(entry);
    for (auto &[key, entry] : descStore)
      if (entry.descriptor)
        Api::descDestroy(entry.descriptor);
    if (preference)
      Api::prefDestroy(preference);
    if (ltHandle)
      Api::ltDestroy(ltHandle);
    if (handle)
      Api::blasDestroy(handle);
    // a destructor can neither throw nor recover: report a failed free and go
    // on
    if (deviceWorkspace) {
      const auto status = Api::rtFree(deviceWorkspace);
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

  static bool isFp8(decltype(Api::RealF32) elementType) {
    return elementType == Api::RealF8E4M3 || elementType == Api::RealF8E5M2;
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
    typename Api::Layout layoutA = nullptr, layoutB = nullptr,
                         layoutC = nullptr;
    typename Api::HeuristicResult heuristicResult{};
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
    typename Api::Layout layoutA = nullptr, layoutB = nullptr,
                         layoutC = nullptr, layoutD = nullptr;
    typename Api::HeuristicResult heuristicResult{};
    bool found = true;
    DescEntry *descEntry = nullptr;
  };
  std::map<QuantBatchKey, QuantBatchedEntry> quantBatchedStore;

  typename Api::Layout
  createBatchedLayout(const std::pair<std::size_t, std::size_t> &shape,
                      std::size_t leadingDimension, long long stride,
                      int batchCount,
                      decltype(Api::RealF32) elementType = Api::RealF32) {
    typename Api::Layout layout = nullptr;
    SOFIEBLAS_CHECK_LT(Api::layoutCreate(&layout, elementType, shape.first,
                                         shape.second, leadingDimension));
    std::int32_t count = batchCount;
    std::int64_t offset = stride;
    SOFIEBLAS_CHECK_LT(Api::layoutSetAttribute(layout, Api::LayoutBatchCount,
                                               &count, sizeof(count)));
    SOFIEBLAS_CHECK_LT(Api::layoutSetAttribute(
        layout, Api::LayoutStridedBatchOffset, &offset, sizeof(offset)));
    return layout;
  }

  static void destroyQuantBatched(QuantBatchedEntry &entry) {
    if (entry.layoutD && entry.layoutD != entry.layoutC)
      Api::layoutDestroy(entry.layoutD);
    for (auto layout : {entry.layoutA, entry.layoutB, entry.layoutC})
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
    const auto apiEpilogue = toApiEpilogue(epilogue);
    QuantBatchKey key{(long long)opA,
                      (long long)opB,
                      (long long)apiEpilogue,
                      (long long)m,
                      (long long)n,
                      (long long)k,
                      strideA,
                      strideB,
                      strideC,
                      (long long)batchCount,
                      (long long)typeAB,
                      (long long)typeD};
    auto entryIterator = quantBatchedStore.find(key);
    if (entryIterator != quantBatchedStore.end()) {
      if (entryIterator->second.found)
        return &entryIterator->second;
      if (!required)
        return nullptr;
      utils::reportQuantBatchedMissing<Api>(m, n, k, batchCount, strideA,
                                            strideB);
    }

    const auto shapeA = layoutKeyA(transa, m, k);
    const auto shapeB = layoutKeyB(transb, k, n);
    const std::pair<std::size_t, std::size_t> shapeC{m, n};
    auto &descEntry = getOrCreateDesc(opA, opB, apiEpilogue, computeType,
                                      scaleType, biasType, typeAB, typeD);
    QuantBatchedEntry newEntry{};
    newEntry.descEntry = &descEntry;
    newEntry.layoutA =
        createBatchedLayout(shapeA, shapeA.first, strideA, batchCount, typeAB);
    newEntry.layoutB =
        createBatchedLayout(shapeB, shapeB.first, strideB, batchCount, typeAB);
    newEntry.layoutC =
        createBatchedLayout(shapeC, m, strideC, batchCount, cTypeFor(typeD));
    newEntry.layoutD =
        cTypeFor(typeD) == typeD
            ? newEntry.layoutC
            : createBatchedLayout(shapeC, m, strideC, batchCount, typeD);
    int returned = 0;
    const auto status =
        Api::getHeuristic(ltHandle, descEntry.descriptor, newEntry.layoutA,
                          newEntry.layoutB, newEntry.layoutC, newEntry.layoutD,
                          preference, 1, &newEntry.heuristicResult, &returned);
    if (required || status != Api::StatusNotSupported)
      SOFIEBLAS_CHECK_LT(status);
    newEntry.found = returned != 0;
    auto *storedEntry = &quantBatchedStore.emplace(key, newEntry).first->second;
    if (newEntry.found)
      return storedEntry;
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
                    std::size_t leadingDimension,
                    decltype(Api::RealF32) elementType = Api::RealF32) {
    if (leadingDimension < shape.first)
      throw std::invalid_argument(std::string("sofieBLAS: leading dimension ") +
                                  std::to_string(leadingDimension) +
                                  " is smaller than the number of rows " +
                                  std::to_string(shape.first) +
                                  " of the matrix.");
    const LayoutKey key{static_cast<int>(elementType), shape.first,
                        shape.second, leadingDimension};
    auto entryIterator = layoutStore.find(key);
    if (entryIterator != layoutStore.end())
      return entryIterator->second;
    typename Api::Layout layout = nullptr;
    SOFIEBLAS_CHECK_LT(Api::layoutCreate(&layout, elementType, shape.first,
                                         shape.second, leadingDimension));
    layoutStore.emplace(key, layout);
    return layout;
  }

  // Sets a pointer attribute of a descriptor, unless that pointer is already
  // set: the entry remembers the last one (field is the member that holds it)
  template <typename TAttr>
  void patchPtr(DescEntry &entry, TAttr attribute,
                const void *DescEntry::*field, const void *pointer) {
    if (entry.*field == pointer)
      return;
    SOFIEBLAS_CHECK_LT(Api::descSetAttribute(entry.descriptor, attribute,
                                             &pointer, sizeof(pointer)));
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
          entry.descriptor, Api::DescBiasBatchStride, &stride, sizeof(stride)));
      entry.biasStride = stride;
    }
  }

  // the pointers to the scales of A, B and D (a null one is left unset)
  void patchScales(DescEntry &entry, const float *scaleA, const float *scaleB,
                   const float *scaleD) {
    if (scaleA)
      patchPtr(entry, Api::DescAScalePtr, &DescEntry::scaleA, scaleA);
    if (scaleB)
      patchPtr(entry, Api::DescBScalePtr, &DescEntry::scaleB, scaleB);
    if (scaleD)
      patchPtr(entry, Api::DescDScalePtr, &DescEntry::scaleD, scaleD);
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
    auto entryIterator = descStore.find(key);
    if (entryIterator != descStore.end())
      return entryIterator->second;

    typename Api::MatmulDesc descriptor = nullptr;
    SOFIEBLAS_CHECK_LT(Api::descCreate(&descriptor, computeType, scaleType));
    SOFIEBLAS_CHECK_LT(Api::descSetAttribute(descriptor, Api::DescTransA,
                                             &transA, sizeof(transA)));
    SOFIEBLAS_CHECK_LT(Api::descSetAttribute(descriptor, Api::DescTransB,
                                             &transB, sizeof(transB)));
    SOFIEBLAS_CHECK_LT(Api::descSetAttribute(descriptor, Api::DescEpilogue,
                                             &epilogue, sizeof(epilogue)));
    // For bias epilogues: set a non-null dummy pointer so the descriptor is
    // valid for the heuristic query, and the bias type the caller declared.
    if (usesBias(epilogue)) {
      const void *placeholderPtr = deviceWorkspace;
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(descriptor, Api::DescBiasPtr,
                                               &placeholderPtr,
                                               sizeof(placeholderPtr)));
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(
          descriptor, Api::DescBiasDataType, &biasType, sizeof(biasType)));
    }
    // FP8: placeholder scale pointers, so the algorithm search sees a scaled
    // multiply; the real ones are set on every call
    if (typeAB == Api::RealF8E4M3) {
      const void *placeholderPtr = deviceWorkspace;
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(descriptor, Api::DescAScalePtr,
                                               &placeholderPtr,
                                               sizeof(placeholderPtr)));
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(descriptor, Api::DescBScalePtr,
                                               &placeholderPtr,
                                               sizeof(placeholderPtr)));
      if (isFp8(typeD))
        SOFIEBLAS_CHECK_LT(Api::descSetAttribute(descriptor, Api::DescDScalePtr,
                                                 &placeholderPtr,
                                                 sizeof(placeholderPtr)));
    }
    // alpha as a device vector with one value per row of D. Beta is always 0
    // here, but it is passed as a host value: cuBLASLt has column-major int8
    // kernels for the BETA_HOST mode and none for BETA_ZERO (which needs the
    // COL32 layouts of the older int8 kernels)
    if constexpr (requires { Api::PtrModeAlphaDeviceVectorBetaHost; }) {
      if (alphaVector) {
        std::int32_t mode = Api::PtrModeAlphaDeviceVectorBetaHost;
        SOFIEBLAS_CHECK_LT(Api::descSetAttribute(descriptor, Api::DescPtrMode,
                                                 &mode, sizeof(mode)));
      }
    }
    DescEntry entry;
    entry.descriptor = descriptor;
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
    auto entryIterator = algoCache.find(key);
    if (entryIterator != algoCache.end()) {
      if (algoCacheLimit)
        lruOrder.splice(lruOrder.begin(), lruOrder,
                        entryIterator->second.lruPosition);
      if (entryIterator->second.found)
        return &entryIterator->second;
      if (!required)
        return nullptr;
      utils::reportAlgoMissing<Api>(transA, transB, epilogue, shapeA, shapeB);
    }

    auto &descEntry =
        getOrCreateDesc(transA, transB, epilogue, computeType, scaleType,
                        biasType, typeAB, typeD, alphaVector);
    auto layoutA = getOrCreateLayout(shapeA, ldA ? ldA : shapeA.first, typeAB);
    auto layoutB = getOrCreateLayout(shapeB, ldB ? ldB : shapeB.first, typeAB);
    auto layoutC = getOrCreateLayout(shapeC, shapeC.first, cTypeFor(typeD));
    auto layoutD = cTypeFor(typeD) == typeD
                       ? layoutC
                       : getOrCreateLayout(shapeC, shapeC.first, typeD);
    typename Api::HeuristicResult heuristicResult{};
    int returnedResults = 0;
    const auto status = Api::getHeuristic(
        ltHandle, descEntry.descriptor, layoutA, layoutB, layoutC, layoutD,
        preference, 1, &heuristicResult, &returnedResults);
    if (required || status != Api::StatusNotSupported)
      SOFIEBLAS_CHECK_LT(status);
    if (returnedResults == 0 && required)
      utils::reportAlgoMissing<Api>(transA, transB, epilogue, shapeA, shapeB);
    CacheEntry entry{heuristicResult, {}, returnedResults != 0};
    entry.descEntry = &descEntry;
    entry.layoutA = layoutA;
    entry.layoutB = layoutB;
    entry.layoutC = layoutC;
    entry.layoutD = layoutD;
    auto insertedEntry = algoCache.emplace(key, entry).first;
    if (algoCacheLimit) {
      lruOrder.push_front(key);
      insertedEntry->second.lruPosition = lruOrder.begin();
      while (algoCache.size() > algoCacheLimit) {
        algoCache.erase(lruOrder.back());
        lruOrder.pop_back();
      }
    }
    return returnedResults != 0 ? &insertedEntry->second : nullptr;
  }

  // the cached descriptor, algorithm and layouts of a multiply
  struct MatmulSetup {
    DescEntry *descEntry = nullptr;
    typename Api::HeuristicResult *algo = nullptr;
    typename Api::Layout layoutA = nullptr, layoutB = nullptr,
                         layoutC = nullptr, layoutD = nullptr;
  };

  // Looks up (or computes) the algorithm and the descriptor of a multiply and
  // patches the real bias pointer into the cached descriptor. The algorithm is
  // null when the library has no kernel for the configuration.
  MatmulSetup
  prepareMatmul(typename Api::Operation transA, typename Api::Operation transB,
                typename Api::Epilogue epilogue, const void *biasPtr,
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
    if (biasPtr && usesBias(epilogue))
      patchPtr(*entry->descEntry, Api::DescBiasPtr, &DescEntry::bias, biasPtr);
    return {entry->descEntry, &entry->heuristicResult, entry->layoutA,
            entry->layoutB,   entry->layoutC,          entry->layoutD};
  }

  // the matmul call of a prepared multiply
  bool launchMatmul(const MatmulSetup &setup, const void *alpha, const void *A,
                    const void *B, const void *beta, const void *inputC,
                    void *outputD, bool required) {
    // C and D at one address need one layout; a C of another type than D
    // (FP8 D, beta 0) is never read, so it is left out
    if (setup.layoutD != setup.layoutC)
      inputC = nullptr;
    const auto status = Api::matmul(
        ltHandle, setup.descEntry->descriptor, alpha, A, setup.layoutA, B,
        setup.layoutB, beta, inputC, setup.layoutC, outputD, setup.layoutD,
        &setup.algo->algo, deviceWorkspace, workspaceSize, stream);
    if (required)
      SOFIEBLAS_CHECK_LT(status);
    return status == Api::StatusSuccess;
  }

  bool executeMatmul(typename Api::Operation transA,
                     typename Api::Operation transB,
                     typename Api::Epilogue epilogue, const void *alpha,
                     const void *A, const void *B, const void *beta,
                     const void *inputC, void *outputD, const void *biasPtr,
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
        transA, transB, epilogue, biasPtr, shapeA, shapeB, shapeC, ldA, ldB,
        computeType, scaleType, typeAB, typeD, biasType, alphaVector, required);
    if (!setup.algo)
      return false;
    return launchMatmul(setup, alpha, A, B, beta, inputC, outputD, required);
  }

  // executeMatmul for the quantized multiplies that scale: the device pointers
  // to the scales of A, B and D (a null one is left unset) are patched into
  // the cached descriptor on every call
  bool executeMatmulWithScale(
      typename Api::Operation transA, typename Api::Operation transB,
      typename Api::Epilogue epilogue, const void *alpha, const void *A,
      const void *B, const void *beta, const void *inputC, void *outputD,
      const void *biasPtr, const std::pair<std::size_t, std::size_t> &shapeA,
      const std::pair<std::size_t, std::size_t> &shapeB,
      const std::pair<std::size_t, std::size_t> &shapeC, std::size_t ldA,
      std::size_t ldB, decltype(Api::ComputeF32) computeType,
      decltype(Api::RealF32) scaleType, decltype(Api::RealF32) typeAB,
      decltype(Api::RealF32) typeD, decltype(Api::RealF32) biasType,
      const float *scaleA, const float *scaleB, const float *scaleD,
      bool alphaVector = false, bool required = true) {
    const auto setup = prepareMatmul(
        transA, transB, epilogue, biasPtr, shapeA, shapeB, shapeC, ldA, ldB,
        computeType, scaleType, typeAB, typeD, biasType, alphaVector, required);
    if (!setup.algo)
      return false;
    patchScales(*setup.descEntry, scaleA, scaleB, scaleD);
    return launchMatmul(setup, alpha, A, B, beta, inputC, outputD, required);
  }

  static GemmMode extendedEpilogueMode(GemmMode plainGemm) {
    return plainGemm == GemmMode::Vendor ? GemmMode::Vendor_withExtendedEpilogue
                                         : GemmMode::Native;
  }

  const GemmModeByEpilogue &fp8Modes(decltype(Api::RealF32) typeD) const {
    if (typeD == Api::RealF16)
      return detectedCapabilities.fp8Fp16;
    if (typeD == Api::RealBF16)
      return detectedCapabilities.fp8Bf16;
    if (typeD == Api::RealF8E4M3)
      return detectedCapabilities.fp8Fp8;
    if (typeD == Api::RealF8E5M2)
      return detectedCapabilities.fp8E5m2;
    return detectedCapabilities.fp8Fp32;
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
    for (int batchIdx = 0; batchIdx < batchCount; ++batchIdx)
      if (!executeMatmul(charToTranspose(transa), charToTranspose(transb),
                         Api::EpilogueDefault, &one, A + batchIdx * strideA,
                         B + batchIdx * strideB, &zero, C + batchIdx * strideC,
                         C + batchIdx * strideC, nullptr,
                         layoutKeyA(transa, m, k), layoutKeyB(transb, k, n),
                         {m, n}, 0, 0, compute, typeSum, typeAB, typeSum,
                         Api::RealF32, false, false))
        return false;
    return true;
  }

  struct SafeVendor {
    using Half = __half;
    BlasLtBase &blasLt;

    bool int8Int32Available() const {
      return blasLt.detectedCapabilities.int8Int32 == GemmMode::Vendor;
    }
    bool fp16Fp32Available() const {
      return blasLt.detectedCapabilities.fp16Fp32 == GemmMode::Vendor;
    }
    bool int8Int32(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, const std::int8_t *A, long long strideA,
                   const std::int8_t *B, long long strideB, std::int32_t *C,
                   long long strideC, int batchCount) {
      return int8Int32Available() &&
             blasLt.vendorSum(transa, transb, m, n, k, A, strideA, B, strideB,
                              C, strideC, batchCount);
    }
    bool fp16Fp32(char transa, char transb, unsigned int m, unsigned int n,
                  unsigned int k, const Half *A, long long strideA,
                  const Half *B, long long strideB, float *C, long long strideC,
                  int batchCount) {
      return fp16Fp32Available() &&
             blasLt.vendorSum(transa, transb, m, n, k, A, strideA, B, strideB,
                              C, strideC, batchCount);
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
    const auto apiEpilogue = toApiEpilogue(epilogue);
    const auto shapeA = layoutKeyA(transa, m, k);
    const auto shapeB = layoutKeyB(transb, k, n);
    if (fp8Modes(typeD)[epilogue] == GemmMode::Vendor) {
      const auto setup =
          prepareMatmul(opA, opB, apiEpilogue, bias, shapeA, shapeB, {m, n}, 0,
                        0, Api::ComputeF32, Api::RealF32, Api::RealF8E4M3,
                        typeD, Api::RealBF16, false, false);
      if (setup.algo) {
        const float beta = 0.f;
        patchScales(*setup.descEntry, scaleA, scaleB, scaleD);
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
            detectedCapabilities.fp8Fp32[epilogue] == GemmMode::Vendor
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
    const auto apiEpilogue = toApiEpilogue(epilogue);
    const auto shapeA = layoutKeyA(transa, m, k);
    const auto shapeB = layoutKeyB(transb, k, n);
    const bool vector = alphaVector != nullptr;
    if ((vector
             ? detectedCapabilities.int8Int8VectorAlpha
             : detectedCapabilities.int8Int8)[epilogue] == GemmMode::Vendor) {
      const auto setup =
          prepareMatmul(opA, opB, apiEpilogue, bias, shapeA, shapeB, {m, n}, 0,
                        0, Api::ComputeI32, Api::RealF32, Api::RealI8,
                        Api::RealI8, Api::RealF32, vector, false);
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
            detectedCapabilities.int8Int8[epilogue] == GemmMode::Vendor
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

  static float applyEpilogue(float value, float bias, Epilogue epilogue) {
    if (epilogue == Epilogue::Bias || epilogue == Epilogue::ReluBias ||
        epilogue == Epilogue::GeluBias)
      value += bias;
    if (epilogue == Epilogue::Relu || epilogue == Epilogue::ReluBias)
      value = value > 0.f ? value : 0.f;
    else if (epilogue == Epilogue::GeluBias || epilogue == Epilogue::Gelu)
      value *=
          0.5f * (1.f + std::tanh(0.7978845608028654f *
                                  (value + 0.044715f * value * value * value)));
    return value;
  }

  void requireQuantization() const {
    if (!quantization)
      throw std::logic_error("sofieBLAS: the quantized methods need a backend "
                             "constructed with {.quantization = true}");
  }

  void enableQuantization(bool forceNative, bool forceVendor) {
    if (forceNative && forceVendor)
      throw std::invalid_argument("sofieBLAS: BlasOptions::forceNative and "
                                  "forceVendor are mutually exclusive");
    quantization = true;
    if (forceNative)
      return;
    if (forceVendor) {
      detectedCapabilities.int8Int32 = GemmMode::Vendor;
      detectedCapabilities.fp16Fp32 = GemmMode::Vendor;
      for (auto *table :
           {&detectedCapabilities.int8Int8,
            &detectedCapabilities.int8Int8VectorAlpha,
            &detectedCapabilities.fp8Fp32, &detectedCapabilities.fp8Fp16,
            &detectedCapabilities.fp8Bf16, &detectedCapabilities.fp8Fp8,
            &detectedCapabilities.fp8E5m2})
        table->fill(GemmMode::Vendor);
      if constexpr (!(requires { Api::PtrModeAlphaDeviceVectorBetaHost; }))
        detectedCapabilities.int8Int8VectorAlpha.fill(
            GemmMode::Vendor_withExtendedEpilogue);
      return;
    }
    detectCapabilities();
  }

  void detectCapabilities() {
    constexpr std::size_t m = 32, n = 32, k = 32;
    const std::pair<std::size_t, std::size_t> shapeA{k, m}, shapeB{k, n},
        shapeC{m, n};
    constexpr Epilogue epilogues[] = {Epilogue::Default,  Epilogue::Bias,
                                      Epilogue::ReluBias, Epilogue::GeluBias,
                                      Epilogue::Relu,     Epilogue::Gelu};
    const auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto upload = [&](const auto &hostVector) {
      using T = typename std::decay_t<decltype(hostVector)>::value_type;
      auto deviceBuffer = alpaka::allocAsyncBufIfSupported<T, std::size_t>(
          m_queue, hostVector.size());
      alpaka::memcpy(
          m_queue, deviceBuffer,
          alpaka::createView(host, hostVector.data(), hostVector.size()));
      alpaka::wait(m_queue);
      return deviceBuffer;
    };
    auto download = [&](auto &deviceBuffer, auto &hostVector) {
      auto view =
          alpaka::createView(host, hostVector.data(), hostVector.size());
      alpaka::memcpy(m_queue, view, deviceBuffer);
      alpaka::wait(m_queue);
    };
    auto reset = [&](auto &deviceBuffer, const auto &hostVector) {
      alpaka::memcpy(
          m_queue, deviceBuffer,
          alpaka::createView(host, hostVector.data(), hostVector.size()));
    };

    std::vector<int> inputA(k * m), inputB(k * n), hostReferenceProduct(m * n);
    std::vector<float> alphas(m), bias(m);
    for (std::size_t i = 0; i < inputA.size(); ++i)
      inputA[i] = static_cast<int>(i * 5 % 7) - 3;
    for (std::size_t i = 0; i < inputB.size(); ++i)
      inputB[i] = static_cast<int>(i * 3 % 5) - 2;
    for (std::size_t i = 0; i < m; ++i) {
      alphas[i] = 0.125f * static_cast<float>(i % 4 + 2);
      bias[i] = -4.f + 0.5f * static_cast<float>(i % 16);
    }
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t p = 0; p < k; ++p)
          hostReferenceProduct[j * m + i] +=
              inputA[i * k + p] * inputB[j * k + p];

    std::vector<std::int8_t> inputA8(inputA.begin(), inputA.end()),
        inputB8(inputB.begin(), inputB.end());
    const std::vector<std::int32_t> poison32(m * n, -77);
    std::vector<std::int32_t> actual32(m * n);
    auto deviceA8 = upload(inputA8);
    auto deviceB8 = upload(inputB8);
    auto deviceC32 = upload(poison32);
    {
      const std::int32_t one = 1, zero = 0;
      const bool vendorRan = executeMatmul(
          Api::OpT, Api::OpN, Api::EpilogueDefault, &one,
          alpaka::getPtrNative(deviceA8), alpaka::getPtrNative(deviceB8), &zero,
          alpaka::getPtrNative(deviceC32), alpaka::getPtrNative(deviceC32),
          nullptr, shapeA, shapeB, shapeC, 0, 0, Api::ComputeI32, Api::RealI32,
          Api::RealI8, Api::RealI32, Api::RealF32, false, false);
      if (vendorRan)
        download(deviceC32, actual32);
      detectedCapabilities.int8Int32 =
          vendorRan && std::equal(actual32.begin(), actual32.end(),
                                  hostReferenceProduct.begin())
              ? GemmMode::Vendor
              : GemmMode::Native;
    }

    std::vector<__half> inputAHalf(inputA.size()), inputBHalf(inputB.size());
    for (std::size_t i = 0; i < inputA.size(); ++i)
      inputAHalf[i] = static_cast<__half>(static_cast<float>(inputA[i]));
    for (std::size_t i = 0; i < inputB.size(); ++i)
      inputBHalf[i] = static_cast<__half>(static_cast<float>(inputB[i]));
    const std::vector<float> poisonF(m * n, -77.f);
    std::vector<float> actualF(m * n);
    auto deviceAHalf = upload(inputAHalf);
    auto deviceBHalf = upload(inputBHalf);
    auto deviceCFloat = upload(poisonF);
    {
      const float one = 1.f, zero = 0.f;
      const bool vendorRan = executeMatmul(
          Api::OpT, Api::OpN, Api::EpilogueDefault, &one,
          alpaka::getPtrNative(deviceAHalf), alpaka::getPtrNative(deviceBHalf),
          &zero, alpaka::getPtrNative(deviceCFloat),
          alpaka::getPtrNative(deviceCFloat), nullptr, shapeA, shapeB, shapeC,
          0, 0, Api::ComputeF32, Api::RealF32, Api::RealF16, Api::RealF32,
          Api::RealF32, false, false);
      bool matchesReference = vendorRan;
      if (vendorRan) {
        download(deviceCFloat, actualF);
        for (std::size_t element = 0; element < m * n && matchesReference;
             ++element)
          matchesReference =
              std::abs(actualF[element] -
                       static_cast<float>(hostReferenceProduct[element])) <=
              1e-3f * (1.f + std::abs(static_cast<float>(
                                 hostReferenceProduct[element])));
      }
      detectedCapabilities.fp16Fp32 =
          matchesReference ? GemmMode::Vendor : GemmMode::Native;
    }

    const std::vector<std::int8_t> poison8(m * n, std::int8_t{-77});
    std::vector<std::int8_t> actual8(m * n);
    auto deviceAlphas = upload(alphas);
    auto deviceBias = upload(bias);
    auto deviceC8 = upload(poison8);
    for (bool vector : {false, true}) {
      auto &table = vector ? detectedCapabilities.int8Int8VectorAlpha
                           : detectedCapabilities.int8Int8;
      table.fill(extendedEpilogueMode(detectedCapabilities.int8Int32));
      if constexpr (!(requires { Api::PtrModeAlphaDeviceVectorBetaHost; }))
        if (vector)
          continue;
      for (Epilogue epilogue : epilogues) {
        reset(deviceC8, poison8);
        const float alpha = 0.5f, beta = 0.f;
        if (!executeMatmul(
                Api::OpT, Api::OpN, toApiEpilogue(epilogue),
                vector ? static_cast<const void *>(
                             alpaka::getPtrNative(deviceAlphas))
                       : static_cast<const void *>(&alpha),
                alpaka::getPtrNative(deviceA8), alpaka::getPtrNative(deviceB8),
                &beta, alpaka::getPtrNative(deviceC8),
                alpaka::getPtrNative(deviceC8),
                alpaka::getPtrNative(deviceBias), shapeA, shapeB, shapeC, 0, 0,
                Api::ComputeI32, Api::RealF32, Api::RealI8, Api::RealI8,
                Api::RealF32, vector, false))
          continue;
        download(deviceC8, actual8);
        const bool geluEpilogue =
            epilogue == Epilogue::GeluBias || epilogue == Epilogue::Gelu;
        bool matchesReference = true;
        for (std::size_t element = 0; element < m * n && matchesReference;
             ++element) {
          const std::size_t i = element % m;
          const float unrounded = applyEpilogue(
              (vector ? alphas[i] : alpha) *
                  static_cast<float>(hostReferenceProduct[element]),
              bias[i], epilogue);
          const int expected = static_cast<int>(
              std::min(127.f, std::max(-128.f, std::nearbyint(unrounded))));
          const int difference =
              std::abs(static_cast<int>(actual8[element]) - expected);
          const float distanceToTie =
              std::abs(unrounded - std::floor(unrounded) - 0.5f);
          matchesReference =
              difference == 0 ||
              (geluEpilogue && difference == 1 && distanceToTie > 0.f &&
               distanceToTie <= 1e-3f);
        }
        if (matchesReference)
          table[epilogue] = GemmMode::Vendor;
      }
    }

    using Fp8 = typename Api::Fp8E4M3;
    using Bf16 = typename Api::Bf16;
    std::vector<Fp8> inputAFp8(inputA.size()), inputBFp8(inputB.size());
    for (std::size_t i = 0; i < inputA.size(); ++i)
      inputAFp8[i] = static_cast<Fp8>(static_cast<float>(inputA[i]));
    for (std::size_t i = 0; i < inputB.size(); ++i)
      inputBFp8[i] = static_cast<Fp8>(static_cast<float>(inputB[i]));
    std::vector<Bf16> biasBf16(m);
    for (std::size_t i = 0; i < m; ++i)
      biasBf16[i] = static_cast<Bf16>(bias[i]);
    const std::vector<float> scales{0.5f, 2.f};
    auto deviceAFp8 = upload(inputAFp8);
    auto deviceBFp8 = upload(inputBFp8);
    auto deviceBiasBf16 = upload(biasBf16);
    auto deviceScales = upload(scales);
    const float *scaleA = alpaka::getPtrNative(deviceScales);
    const float *scaleB = scaleA + 1;
    for (auto *table :
         {&detectedCapabilities.fp8Fp32, &detectedCapabilities.fp8Fp16,
          &detectedCapabilities.fp8Bf16, &detectedCapabilities.fp8Fp8,
          &detectedCapabilities.fp8E5m2})
      table->fill(extendedEpilogueMode(detectedCapabilities.fp16Fp32));
    for (Epilogue epilogue : epilogues) {
      reset(deviceCFloat, poisonF);
      const float alpha = 0.25f, beta = 0.f;
      if (!executeMatmulWithScale(
              Api::OpT, Api::OpN, toApiEpilogue(epilogue), &alpha,
              alpaka::getPtrNative(deviceAFp8),
              alpaka::getPtrNative(deviceBFp8), &beta,
              alpaka::getPtrNative(deviceCFloat),
              alpaka::getPtrNative(deviceCFloat),
              alpaka::getPtrNative(deviceBiasBf16), shapeA, shapeB, shapeC, 0,
              0, Api::ComputeF32, Api::RealF32, Api::RealF8E4M3, Api::RealF32,
              Api::RealBF16, scaleA, scaleB, nullptr, false, false))
        continue;
      download(deviceCFloat, actualF);
      bool matchesReference = true;
      for (std::size_t element = 0; element < m * n && matchesReference;
           ++element) {
        const float expected =
            applyEpilogue(alpha * scales[0] * scales[1] *
                              static_cast<float>(hostReferenceProduct[element]),
                          bias[element % m], epilogue);
        matchesReference = std::abs(actualF[element] - expected) <=
                           1e-3f * (1.f + std::abs(expected));
      }
      if (!matchesReference)
        continue;
      detectedCapabilities.fp8Fp32[epilogue] = GemmMode::Vendor;
      const auto apiEpilogue = toApiEpilogue(epilogue);
      for (auto [typeD, table] :
           {std::pair{Api::RealF16, &detectedCapabilities.fp8Fp16},
            std::pair{Api::RealBF16, &detectedCapabilities.fp8Bf16},
            std::pair{Api::RealF8E4M3, &detectedCapabilities.fp8Fp8},
            std::pair{Api::RealF8E5M2, &detectedCapabilities.fp8E5m2}})
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
  void executeQuantBatched(QuantBatchedEntry &batchedEntry, const void *alpha,
                           const void *A, const void *B, const void *beta,
                           const void *bias, void *C, Epilogue epilogue,
                           const float *scaleA = nullptr,
                           const float *scaleB = nullptr) {
    auto &descEntry = *batchedEntry.descEntry;
    if (bias && usesBias(toApiEpilogue(epilogue))) {
      patchPtr(descEntry, Api::DescBiasPtr, &DescEntry::bias, bias);
      patchBiasStride(descEntry, 0);
    }
    patchScales(descEntry, scaleA, scaleB, nullptr);
    SOFIEBLAS_CHECK_LT(Api::matmul(
        ltHandle, descEntry.descriptor, alpha, A, batchedEntry.layoutA, B,
        batchedEntry.layoutB, beta, C, batchedEntry.layoutC, C,
        batchedEntry.layoutD, &batchedEntry.heuristicResult.algo,
        deviceWorkspace, workspaceSize, stream));
  }
};
