// Shared implementation of the cuBLASLt and hipBLASLt backends. The two
// vendor APIs have the same shape under different names, so the backend is
// written once against an Api table. A vendor header defines that table
// (the types, constants and functions of its library), defines the check
// macros SOFIEBLAS_CHECK_LT and SOFIEBLAS_CHECK_RT, includes the vendor and
// standard headers (<cstdlib>, <functional>, <iostream>, <list>,
// <stdexcept>, <string>, <unordered_map>, <utility>, alpaka), and then
// includes this file.

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

struct DescKey {
  int transA; // backend transpose enum encoded as int
  int transB;
  int epilogue; // backend epilogue enum encoded as int
  int computeType;
  int scaleType;
  bool operator==(const DescKey &o) const noexcept {
    return transA == o.transA && transB == o.transB && epilogue == o.epilogue &&
           computeType == o.computeType && scaleType == o.scaleType;
  }
};

struct DescKeyHash {
  std::size_t operator()(const DescKey &k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.transA) * 97u +
                    static_cast<std::size_t>(k.transB) * 31u +
                    static_cast<std::size_t>(k.epilogue);
    h = h * 31u + static_cast<std::size_t>(k.computeType);
    h = h * 31u + static_cast<std::size_t>(k.scaleType);
    return h ^ (h >> 16);
  }
};

struct AlgoKey {
  DescKey dk;
  std::size_t rowsA, colsA; // physical dimensions of A in layoutStore
  std::size_t rowsB, colsB; // physical dimensions of B in layoutStore
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

template <class Api> class BlasLt {
  typename Api::Handle ltHandle = nullptr;
  typename Api::BlasHandle handle = nullptr;
  typename Api::Preference preference = nullptr;
  void *d_workspace = nullptr;
  size_t workspaceSize = 1u << 25; // 32 MB
  typename Api::Stream stream = nullptr;

  std::unordered_map<LayoutKey, typename Api::Layout, LayoutKeyHash>
      layoutStore;

  std::unordered_map<DescKey, typename Api::MatmulDesc, DescKeyHash> descStore;

  // One cache entry per exact GEMM configuration: the heuristic result to
  // reuse, plus this entry's position in the recency list so a hit can mark
  // itself most-recently-used in O(1). The position is only maintained when a
  // cache limit is set; with no limit the list stays empty.
  struct CacheEntry {
    typename Api::HeuristicResult h{};
    std::list<AlgoKey>::iterator lru{};
  };
  std::unordered_map<AlgoKey, CacheEntry, AlgoKeyHash> algoCache;
  // entries ordered most- to least-recently used; drives eviction
  std::list<AlgoKey> lruOrder;
  // 0 = unbounded
  std::size_t algoCacheLimit = 0;

public:
  std::size_t algoCacheSize() const { return algoCache.size(); }

  BlasLt(const BlasLt &) = delete;
  BlasLt &operator=(const BlasLt &) = delete;
  BlasLt(BlasLt &&) = delete;
  BlasLt &operator=(BlasLt &&) = delete;

  BlasLt(typename Api::Queue &queue, std::size_t cacheLimit = 0)
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

  ~BlasLt() {
    for (auto &[key, layout] : layoutStore)
      if (layout)
        Api::layoutDestroy(layout);
    for (auto &[key, desc] : descStore)
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

  inline typename Api::Epilogue toApiEpilogue(Epilogue epilogue) {
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

  // Registers a call site's construction-time shape: creates the three matrix
  // layouts and resolves the multiply algorithm for them up front, so the
  // first call at this shape finds everything cached.
  void addOperationConfig(std::size_t m, std::size_t n, std::size_t k,
                          std::size_t lda, std::size_t ldb, std::size_t ldc,
                          char transa, char transb, Epilogue epilogue) {
    const auto shapeA = layoutKeyA(transa, m, k);
    const auto shapeB = layoutKeyB(transb, k, n);
    const std::pair<std::size_t, std::size_t> shapeC{m, n};
    getOrCreateLayout(shapeA, lda);
    getOrCreateLayout(shapeB, ldb);
    getOrCreateLayout(shapeC, ldc);

    getOrComputeAlgo(charToTranspose(transa), charToTranspose(transb),
                     toApiEpilogue(epilogue), shapeA, shapeB, shapeC);
  }

  // Each multiply variant comes as one generic overload, where A, B, bias and
  // C are any alpaka buffers or views (anything alpaka::getPtrNative
  // accepts), and one raw device-pointer overload, which generated code
  // calls.
  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, TA const &A, TB const &B,
                   float beta, TBias &bias, TC &C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueBias, &alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), &beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, T const *B,
                   float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueBias, &alpha, A, B, &beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, TA const &A, TB const &B,
                       float beta, TBias &bias, TC &C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueReluBias, &alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), &beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueReluBias, &alpha, A, B, &beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, TA const &A, TB const &B,
                       float beta, TBias &bias, TC &C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueGeluBias, &alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), &beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueGeluBias, &alpha, A, B, &beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename TA, typename TB, typename TC>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, TA const &A, TB const &B,
                     float beta, TC &C) {
    auto *c = alpaka::getPtrNative(C);
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueDefault, &alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), &beta, c, c, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, T const *B,
                     float beta, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueDefault, &alpha, A, B, &beta, C, C, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  inline void gemmStridedBatched(char transa, char transb, int m, int n, int k,
                                 float alpha, const float *A, int lda,
                                 long long strideA, const float *B, int ldb,
                                 long long strideB, float beta, float *C,
                                 int ldc, long long strideC, int batchCount) {
    SOFIEBLAS_CHECK_LT(Api::sgemmStridedBatched(
        handle, charToTranspose(transa), charToTranspose(transb), m, n, k,
        &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC,
        batchCount));
  }

  template <typename TA, typename TB, typename TC>
  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k,
                         TA const &A, TB const &B, TC &C) {
    const int32_t alpha = 1, beta = 0;
    auto *c = alpaka::getPtrNative(C);
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueDefault, &alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), &beta, c, c, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n},
                  Api::ComputeI32, Api::RealI32, Api::RealI8, Api::RealI32);
  }

  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k,
                         const int8_t *A, const int8_t *B, int32_t *C) {
    const int32_t alpha = 1, beta = 0;
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueDefault, &alpha, A, B, &beta, C, C, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n},
                  Api::ComputeI32, Api::RealI32, Api::RealI8, Api::RealI32);
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k, float alpha,
                         TA const &A, TB const &B, TBias &bias, TC &C,
                         Epilogue epilogue) {
    const float beta = 0.f;
    auto *c = alpaka::getPtrNative(C);
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  toApiEpilogue(epilogue), &alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), &beta, c, c,
                  alpaka::getPtrNative(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n}, Api::ComputeI32,
                  Api::RealF32, Api::RealI8, Api::RealI8);
  }

  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k, float alpha,
                         const int8_t *A, const int8_t *B, const float *bias,
                         int8_t *C, Epilogue epilogue) {
    const float beta = 0.f;
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  toApiEpilogue(epilogue), &alpha, A, B, &beta, C, C, bias,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n},
                  Api::ComputeI32, Api::RealF32, Api::RealI8, Api::RealI8);
  }

  // Checks whether the library has a kernel for the int8-output int8Matmul
  // above (int8 A, B and C, float alpha and bias) with these transposes,
  // sizes and epilogue. Returns false instead of ending the process when it
  // has none, so the caller can take another path, such as the int32-output
  // int8Matmul.
  inline bool int8MatmulSupported(char transa, char transb, unsigned int m,
                                  unsigned int n, unsigned int k,
                                  Epilogue epilogue) {
    return getOrComputeAlgo(charToTranspose(transa), charToTranspose(transb),
                            toApiEpilogue(epilogue), layoutKeyA(transa, m, k),
                            layoutKeyB(transb, k, n), {m, n}, Api::ComputeI32,
                            Api::RealF32, Api::RealI8, Api::RealI8,
                            /*exitOnNoAlgorithm=*/false) != nullptr;
  }

private:
  typename Api::Queue m_queue;

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

  // Returns the layout describing a (rows, cols) matrix, creating and caching
  // it on first use. Every caller passes ld = rows (dense column-major).
  typename Api::Layout
  getOrCreateLayout(const std::pair<std::size_t, std::size_t> &shape,
                    std::size_t ld, decltype(Api::RealF32) type = Api::RealF32) {
    const LayoutKey key{static_cast<int>(type), shape.first, shape.second};
    auto it = layoutStore.find(key);
    if (it != layoutStore.end())
      return it->second;
    typename Api::Layout layout = nullptr;
    SOFIEBLAS_CHECK_LT(Api::layoutCreate(&layout, type, shape.first,
                                         shape.second, ld));
    layoutStore.emplace(key, layout);
    return layout;
  }

  typename Api::MatmulDesc &
  getOrCreateDesc(typename Api::Operation transA, typename Api::Operation transB,
                  typename Api::Epilogue epilogue,
                  decltype(Api::ComputeF32) computeType = Api::ComputeF32,
                  decltype(Api::RealF32) scaleType = Api::RealF32) {
    DescKey key{(int)transA, (int)transB, (int)epilogue, (int)computeType,
                (int)scaleType};
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
    // valid for the heuristic query, and give the bias the scale type (FP8
    // will need BF16 here).
    if (epilogue != Api::EpilogueDefault && epilogue != Api::EpilogueRelu) {
      const void *dummy = d_workspace;
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasPointer,
                                               &dummy, sizeof(dummy)));
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasDataType,
                                               &scaleType, sizeof(scaleType)));
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
                   decltype(Api::ComputeF32) computeType = Api::ComputeF32,
                   decltype(Api::RealF32) scaleType = Api::RealF32,
                   decltype(Api::RealF32) typeAB = Api::RealF32,
                   decltype(Api::RealF32) typeC = Api::RealF32,
                   bool exitOnNoAlgorithm = true) {
    AlgoKey key{{(int)transA, (int)transB, (int)epilogue, (int)computeType,
                 (int)scaleType},
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

    auto &desc =
        getOrCreateDesc(transA, transB, epilogue, computeType, scaleType);
    auto lA = getOrCreateLayout(shapeA, shapeA.first, typeAB);
    auto lB = getOrCreateLayout(shapeB, shapeB.first, typeAB);
    auto lC = getOrCreateLayout(shapeC, shapeC.first, typeC);
    typename Api::HeuristicResult h{};
    int returnedResults = 0;
    const auto status = Api::getHeuristic(ltHandle, desc, lA, lB, lC, lC,
                                          preference, 1, &h, &returnedResults);
    if (!exitOnNoAlgorithm && status == Api::StatusNotSupported)
      return nullptr;
    SOFIEBLAS_CHECK_LT(status);
    if (returnedResults == 0) {
      if (!exitOnNoAlgorithm)
        return nullptr;
      std::cerr << "[sofieBLAS] No suitable " << Api::name
                << " algorithm found for "
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
                     decltype(Api::ComputeF32) computeType = Api::ComputeF32,
                     decltype(Api::RealF32) scaleType = Api::RealF32,
                     decltype(Api::RealF32) typeAB = Api::RealF32,
                     decltype(Api::RealF32) typeC = Api::RealF32) {
    // Retrieve (or lazily compute) the cached algorithm for this shape
    auto *h = getOrComputeAlgo(transA, transB, epilogue, shapeA, shapeB, shapeC,
                               computeType, scaleType, typeAB, typeC);

    // Retrieve the cached descriptor and patch the real bias pointer in-place
    auto &desc =
        getOrCreateDesc(transA, transB, epilogue, computeType, scaleType);
    if (bias_ptr) {
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasPointer,
                                               &bias_ptr, sizeof(bias_ptr)));
    }

    auto lA = getOrCreateLayout(shapeA, shapeA.first, typeAB);
    auto lB = getOrCreateLayout(shapeB, shapeB.first, typeAB);
    auto lC = getOrCreateLayout(shapeC, shapeC.first, typeC);
    SOFIEBLAS_CHECK_LT(Api::matmul(ltHandle, desc, alpha, A, lA, B, lB, beta,
                                   D_in, lC, C_out, lC, &h->algo, d_workspace,
                                   workspaceSize, stream));
  }
};
