// Shared implementation of the cuBLASLt and hipBLASLt backends. The two
// vendor APIs have the same shape under different names, so the backend is
// written once against an Api table. A vendor header defines that table
// (the types, constants and functions of its library), defines the check
// macros SOFIEBLAS_CHECK_LT and SOFIEBLAS_CHECK_RT, includes the vendor and
// standard headers (<cstdint>, <cstdlib>, <functional>, <iostream>, <list>,
// <stdexcept>, <string>, <unordered_map>, <utility>, alpaka), and then
// includes this file.

struct LayoutKey {
  int type;
  std::size_t rows, cols;
  int batchCount;
  std::int64_t batchStride;
  bool operator==(const LayoutKey &o) const noexcept {
    return type == o.type && rows == o.rows && cols == o.cols &&
           batchCount == o.batchCount && batchStride == o.batchStride;
  }
};

struct LayoutKeyHash {
  std::size_t operator()(const LayoutKey &k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.type);
    auto mix = [&](std::size_t v) {
      h ^= std::hash<std::size_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) +
           (h >> 2);
    };
    mix(k.rows);
    mix(k.cols);
    mix(static_cast<std::size_t>(k.batchCount));
    mix(static_cast<std::size_t>(k.batchStride));
    return h;
  }
};

struct DescKey {
  int transA; // backend transpose enum encoded as int
  int transB;
  int epilogue; // backend epilogue enum encoded as int
  int compute;  // backend compute-type enum encoded as int
  int scaleType; // backend data-type enum of alpha and beta
  int biasType;  // backend data-type enum of the bias, -1 = library default
  bool operator==(const DescKey &o) const noexcept {
    return transA == o.transA && transB == o.transB &&
           epilogue == o.epilogue && compute == o.compute &&
           scaleType == o.scaleType && biasType == o.biasType;
  }
};

struct DescKeyHash {
  std::size_t operator()(const DescKey &k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.transA) * 97u +
                    static_cast<std::size_t>(k.transB) * 31u +
                    static_cast<std::size_t>(k.epilogue);
    h = h * 131u + static_cast<std::size_t>(k.compute);
    h = h * 131u + static_cast<std::size_t>(k.scaleType);
    h = h * 131u + static_cast<std::size_t>(k.biasType + 1);
    return h ^ (h >> 16);
  }
};

struct AlgoKey {
  DescKey dk;
  LayoutKey a, b, c;
  bool operator==(const AlgoKey &o) const noexcept {
    return dk == o.dk && a == o.a && b == o.b && c == o.c;
  }
};

struct AlgoKeyHash {
  std::size_t operator()(const AlgoKey &k) const noexcept {
    std::size_t h = DescKeyHash{}(k.dk);
    auto mix = [&](std::size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(LayoutKeyHash{}(k.a));
    mix(LayoutKeyHash{}(k.b));
    mix(LayoutKeyHash{}(k.c));
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

  // Registers a call site's construction-time shape: creates the three matrix
  // layouts and resolves the multiply algorithm for them up front, so the
  // first call at this shape finds everything cached.
  void addOperationConfig(std::size_t m, std::size_t n, std::size_t k,
                          std::size_t lda, std::size_t ldb, std::size_t ldc,
                          char transa, char transb, Epilogue epilogue) {
    typename Api::Epilogue apiEpilogue = Api::EpilogueDefault;
    switch (epilogue) {
    case Epilogue::Bias:
      apiEpilogue = Api::EpilogueBias;
      break;
    case Epilogue::ReluBias:
      apiEpilogue = Api::EpilogueReluBias;
      break;
    case Epilogue::GeluBias:
      apiEpilogue = Api::EpilogueGeluBias;
      break;
    case Epilogue::Relu:
      apiEpilogue = Api::EpilogueRelu;
      break;
    case Epilogue::Default:
      break;
    }
    const AlgoKey op{descKey(transa, transb, apiEpilogue),
                     layoutKeyA(transa, m, k), layoutKeyB(transb, k, n),
                     layoutKeyC(m, n)};
    getOrCreateLayout(op.a, lda);
    getOrCreateLayout(op.b, ldb);
    getOrCreateLayout(op.c, ldc);
    getOrComputeAlgo(op);
  }

  // Each multiply variant comes as one generic overload, where A, B, bias and
  // C are any alpaka buffers or views (anything alpaka::getPtrNative
  // accepts), and one raw device-pointer overload, which generated code
  // calls.
  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, TA const &A, TB const &B,
                   float beta, TBias &bias, TC &C) {
    gemm(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
         alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
         alpaka::getPtrNative(C));
  }

  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, T const *B,
                   float beta, T *bias, T *C) {
    executeMatmul({descKey(transa, transb, Api::EpilogueBias),
                   layoutKeyA(transa, m, k), layoutKeyB(transb, k, n),
                   layoutKeyC(m, n)},
                  &alpha, A, B, &beta, bias, C, bias);
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, TA const &A, TB const &B,
                       float beta, TBias &bias, TC &C) {
    gemmrelu(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
             alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
             alpaka::getPtrNative(C));
  }

  template <typename T>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul({descKey(transa, transb, Api::EpilogueReluBias),
                   layoutKeyA(transa, m, k), layoutKeyB(transb, k, n),
                   layoutKeyC(m, n)},
                  &alpha, A, B, &beta, bias, C, bias);
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, TA const &A, TB const &B,
                       float beta, TBias &bias, TC &C) {
    gemmgelu(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
             alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
             alpaka::getPtrNative(C));
  }

  template <typename T>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul({descKey(transa, transb, Api::EpilogueGeluBias),
                   layoutKeyA(transa, m, k), layoutKeyB(transb, k, n),
                   layoutKeyC(m, n)},
                  &alpha, A, B, &beta, bias, C, bias);
  }

  template <typename TA, typename TB, typename TC>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, TA const &A, TB const &B,
                     float beta, TC &C) {
    matmul(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
           alpaka::getPtrNative(B), beta, alpaka::getPtrNative(C));
  }

  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, T const *B,
                     float beta, T *C) {
    executeMatmul({descKey(transa, transb, Api::EpilogueDefault),
                   layoutKeyA(transa, m, k), layoutKeyB(transb, k, n),
                   layoutKeyC(m, n)},
                  &alpha, A, B, &beta, C, C, nullptr);
  }

  // int8 x int8 -> int32: the operands are quantized codes and the result is
  // their exact accumulated dot product, for the caller to scale and
  // requantize. alpha and beta are integers accordingly. A batchCount above
  // 1 runs a strided batch, the strides counted in elements.
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, std::int32_t alpha, const std::int8_t *A,
                     const std::int8_t *B, std::int32_t beta, std::int32_t *C,
                     int batchCount = 1, std::int64_t strideA = 0,
                     std::int64_t strideB = 0, std::int64_t strideC = 0) {
    executeMatmul(
        {descKey(transa, transb, Api::EpilogueDefault, Api::ComputeI32,
                 Api::RealI32),
         layoutKeyA(transa, m, k, Api::RealI8, batchCount, strideA),
         layoutKeyB(transb, k, n, Api::RealI8, batchCount, strideB),
         layoutKeyC(m, n, Api::RealI32, batchCount, strideC)},
        &alpha, A, B, &beta, C, C, nullptr);
  }

  // Whether the library has a kernel for the matmul above at this shape,
  // registering its layouts and algorithm when it does. Unlike the float
  // addOperationConfig this never ends the process: a caller with a
  // fallback reads the answer.
  inline bool addOperationConfig(char transa, char transb, std::size_t m,
                                 std::size_t n, std::size_t k, DataType in,
                                 DataType out, int batchCount = 1,
                                 std::int64_t strideA = 0,
                                 std::int64_t strideB = 0,
                                 std::int64_t strideC = 0) {
    const auto compute = in == DataType::I8 ? Api::ComputeI32 : Api::ComputeF32;
    const auto scaleType = in == DataType::I8 ? Api::RealI32 : Api::RealF32;
    const AlgoKey op{
        descKey(transa, transb, Api::EpilogueDefault, compute, scaleType),
        layoutKeyA(transa, m, k, realType(in), batchCount, strideA),
        layoutKeyB(transb, k, n, realType(in), batchCount, strideB),
        layoutKeyC(m, n, realType(out), batchCount, strideC)};
    return getOrComputeAlgo(op, /*tolerateMissing=*/true) != nullptr;
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

private:
  typename Api::Queue m_queue;

  // Identities of the three operands of D = op(A) * op(B), as stored. A is
  // m x k and B is k x n after the transposes; C and D are m x n.
  static LayoutKey layoutKeyA(char trans, std::size_t m, std::size_t k,
                              decltype(Api::RealF32) type = Api::RealF32,
                              int batchCount = 1,
                              std::int64_t batchStride = 0) {
    const bool n = (trans == 'N' || trans == 'n');
    return {static_cast<int>(type), n ? m : k, n ? k : m, batchCount,
            batchStride};
  }

  static LayoutKey layoutKeyB(char trans, std::size_t k, std::size_t n,
                              decltype(Api::RealF32) type = Api::RealF32,
                              int batchCount = 1,
                              std::int64_t batchStride = 0) {
    const bool nn = (trans == 'N' || trans == 'n');
    return {static_cast<int>(type), nn ? k : n, nn ? n : k, batchCount,
            batchStride};
  }

  static LayoutKey layoutKeyC(std::size_t m, std::size_t n,
                              decltype(Api::RealF32) type = Api::RealF32,
                              int batchCount = 1,
                              std::int64_t batchStride = 0) {
    return {static_cast<int>(type), m, n, batchCount, batchStride};
  }

  static decltype(Api::RealF32) realType(DataType type) {
    switch (type) {
    case DataType::I8:
      return Api::RealI8;
    case DataType::I32:
      return Api::RealI32;
    case DataType::F32:
      break;
    }
    return Api::RealF32;
  }

  DescKey descKey(char transa, char transb, typename Api::Epilogue epilogue,
                  decltype(Api::ComputeF32) compute = Api::ComputeF32,
                  decltype(Api::RealF32) scaleType = Api::RealF32,
                  int biasType = -1) {
    return {static_cast<int>(charToTranspose(transa)),
            static_cast<int>(charToTranspose(transb)),
            static_cast<int>(epilogue),
            static_cast<int>(compute),
            static_cast<int>(scaleType),
            biasType};
  }

  // Returns the layout a key describes, creating and caching it on first
  // use. Every caller passes ld = rows (dense column-major). A batchCount
  // above 1 describes a strided batch.
  typename Api::Layout getOrCreateLayout(const LayoutKey &key,
                                         std::size_t ld) {
    auto it = layoutStore.find(key);
    if (it != layoutStore.end())
      return it->second;
    typename Api::Layout layout = nullptr;
    const auto type = static_cast<decltype(Api::RealF32)>(key.type);
    SOFIEBLAS_CHECK_LT(
        Api::layoutCreate(&layout, type, key.rows, key.cols, ld));
    if (key.batchCount > 1) {
      if (key.batchStride <= 0)
        throw std::invalid_argument(
            "A strided batch needs a positive batch stride.");
      const std::int32_t count = key.batchCount;
      SOFIEBLAS_CHECK_LT(Api::layoutSetAttribute(layout, Api::LayoutBatchCount,
                                                 &count, sizeof(count)));
      SOFIEBLAS_CHECK_LT(Api::layoutSetAttribute(layout, Api::LayoutBatchStride,
                                                 &key.batchStride,
                                                 sizeof(key.batchStride)));
    }
    layoutStore.emplace(key, layout);
    return layout;
  }

  // Returns the descriptor a key describes, creating and caching it on first
  // use. compute is the accumulation type, scaleType the type alpha and beta
  // are given in, biasType the bias vector's type or -1 to leave the library
  // default (the output type).
  typename Api::MatmulDesc &getOrCreateDesc(const DescKey &key) {
    auto it = descStore.find(key);
    if (it != descStore.end())
      return it->second;

    const auto transA = static_cast<typename Api::Operation>(key.transA);
    const auto transB = static_cast<typename Api::Operation>(key.transB);
    const auto epilogue = static_cast<typename Api::Epilogue>(key.epilogue);
    const auto compute = static_cast<decltype(Api::ComputeF32)>(key.compute);
    const auto scaleType = static_cast<decltype(Api::RealF32)>(key.scaleType);

    typename Api::MatmulDesc desc = nullptr;
    SOFIEBLAS_CHECK_LT(Api::descCreate(&desc, compute, scaleType));
    SOFIEBLAS_CHECK_LT(
        Api::descSetAttribute(desc, Api::DescTransA, &transA, sizeof(transA)));
    SOFIEBLAS_CHECK_LT(
        Api::descSetAttribute(desc, Api::DescTransB, &transB, sizeof(transB)));
    SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescEpilogue, &epilogue,
                                             sizeof(epilogue)));
    // For bias epilogues: set a non-null dummy pointer so the descriptor is
    // valid for the heuristic query.
    if (epilogue != Api::EpilogueDefault) {
      const void *dummy = d_workspace;
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasPointer,
                                               &dummy, sizeof(dummy)));
    }
    if (key.biasType >= 0) {
      const auto biasType = static_cast<decltype(Api::RealF32)>(key.biasType);
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(
          desc, Api::DescBiasDataType, &biasType, sizeof(biasType)));
    }
    descStore.emplace(key, desc);
    return descStore.at(key);
  }

  // Returns the algorithm for one operation, querying the heuristic and
  // caching the answer on first use. With tolerateMissing the two ways the
  // library can decline (an unsupported configuration, or no kernel for the
  // shape) return nullptr instead of ending the process, so a caller with a
  // fallback can take it.
  typename Api::HeuristicResult *getOrComputeAlgo(const AlgoKey &key,
                                                  bool tolerateMissing = false) {
    auto it = algoCache.find(key);
    if (it != algoCache.end()) {
      if (algoCacheLimit)
        lruOrder.splice(lruOrder.begin(), lruOrder, it->second.lru);
      return &it->second.h;
    }

    auto &desc = getOrCreateDesc(key.dk);
    auto lA = getOrCreateLayout(key.a, key.a.rows);
    auto lB = getOrCreateLayout(key.b, key.b.rows);
    auto lC = getOrCreateLayout(key.c, key.c.rows);
    typename Api::HeuristicResult h{};
    int returnedResults = 0;
    const auto status = Api::getHeuristic(ltHandle, desc, lA, lB, lC, lC,
                                          preference, 1, &h, &returnedResults);
    if (tolerateMissing && status == Api::StatusNotSupported)
      return nullptr;
    SOFIEBLAS_CHECK_LT(status);
    if (returnedResults == 0) {
      if (tolerateMissing)
        return nullptr;
      std::cerr << "[sofieBLAS] No suitable " << Api::name
                << " algorithm found for "
                << "transA=" << key.dk.transA << " transB=" << key.dk.transB
                << " epilogue=" << key.dk.epilogue << " A=[" << key.a.rows
                << "x" << key.a.cols << "]"
                << " B=[" << key.b.rows << "x" << key.b.cols << "]\n";
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
    return ins->second.h;
  }

  // Runs one operation: alpha and beta point at scalars of the descriptor's
  // scale type, C_in is read when beta is nonzero, D_out receives the result.
  void executeMatmul(const AlgoKey &op, const void *alpha, const void *A,
                     const void *B, const void *beta, const void *C_in,
                     void *D_out, const void *bias_ptr) {
    // Retrieve (or lazily compute) the cached algorithm for this operation
    auto *h = getOrComputeAlgo(op);

    // Retrieve the cached descriptor and patch the real bias pointer in-place
    auto &desc = getOrCreateDesc(op.dk);
    if (bias_ptr) {
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasPointer,
                                               &bias_ptr, sizeof(bias_ptr)));
    }

    auto lA = getOrCreateLayout(op.a, op.a.rows);
    auto lB = getOrCreateLayout(op.b, op.b.rows);
    auto lC = getOrCreateLayout(op.c, op.c.rows);
    SOFIEBLAS_CHECK_LT(Api::matmul(ltHandle, desc, alpha, A, lA, B, lB, beta,
                                   C_in, lC, D_out, lC, &h->algo, d_workspace,
                                   workspaceSize, stream));
  }
};
