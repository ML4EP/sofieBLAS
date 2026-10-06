// Shared implementation of the cuBLASLt and hipBLASLt backends. The two
// vendor APIs have the same shape under different names, so the backend is
// written once against an Api table. A vendor header defines that table
// (the types, constants and functions of its library), defines the check
// macros SOFIEBLAS_CHECK_LT and SOFIEBLAS_CHECK_RT, includes the vendor and
// standard headers and then includes this file.

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

struct LdKey {
  std::size_t rows, cols, ld;
  bool operator==(const LdKey &o) const noexcept {
    return rows == o.rows && cols == o.cols && ld == o.ld;
  }
};

struct LdKeyHash {
  std::size_t operator()(const LdKey &k) const noexcept {
    return PairHash{}({k.rows, k.cols}) ^ (std::hash<std::size_t>{}(k.ld) * 31u);
  }
};

struct DescKey {
  int transA; // backend transpose enum encoded as int
  int transB;
  int epilogue; // backend epilogue enum encoded as int
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
  std::size_t rowsA, colsA; // physical dimensions of A in layoutStore
  std::size_t rowsB, colsB; // physical dimensions of B in layoutStore
  std::size_t ldA = 0, ldB = 0; // leading dimensions of A and B, 0 when dense
  bool operator==(const AlgoKey &o) const noexcept {
    return dk == o.dk && rowsA == o.rowsA && colsA == o.colsA &&
           rowsB == o.rowsB && colsB == o.colsB && ldA == o.ldA &&
           ldB == o.ldB;
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
    mix(k.ldA);
    mix(k.ldB);
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

  std::unordered_map<std::pair<std::size_t, std::size_t>, typename Api::Layout,
                     PairHash, PairEq>
      layoutStore;

  // layouts of the operands with a leading dimension larger than the number of
  // rows (padded or sub-matrix views); the dense ones are in layoutStore
  std::unordered_map<LdKey, typename Api::Layout, LdKeyHash> ldLayoutStore;

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
    for (auto &[key, layout] : ldLayoutStore)
      if (layout)
        Api::layoutDestroy(layout);
    for (auto &[key, entry] : batchedStore) {
      Api::layoutDestroy(entry.lA);
      Api::layoutDestroy(entry.lB);
      Api::layoutDestroy(entry.lC);
    }
    for (auto &[key, layout] : i8LayoutStore)
      if (layout)
        Api::layoutDestroy(layout);
    for (auto &[key, desc] : descStore)
      if (desc)
        Api::descDestroy(desc);
    for (auto &[key, desc] : i8DescStore)
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
                  Api::EpilogueBias, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, T const *B,
                   float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueBias, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, TA const &A, TB const &B,
                       float beta, TBias &bias, TC &C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueReluBias, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueReluBias, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, TA const &A, TB const &B,
                       float beta, TBias &bias, TC &C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueGeluBias, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueGeluBias, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n});
  }

  template <typename TA, typename TB, typename TC>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, TA const &A, TB const &B,
                     float beta, TC &C) {
    auto *c = alpaka::getPtrNative(C);
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueDefault, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), beta, c, c, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, T const *B,
                     float beta, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueDefault, alpha, A, B, beta, C, C, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  // Variants taking the leading dimensions lda and ldb of A and B (column-major,
  // as the physical rows of the matrix: lda >= m for 'N', >= k for 'T'; ldb >= k
  // for 'N', >= n for 'T'). C is dense.
  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, unsigned int lda,
                   T const *B, unsigned int ldb, float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueBias, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n}, lda, ldb);
  }

  template <typename T>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A,
                       unsigned int lda, T const *B, unsigned int ldb,
                       float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueReluBias, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n}, lda, ldb);
  }

  template <typename T>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A,
                       unsigned int lda, T const *B, unsigned int ldb,
                       float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueGeluBias, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n}, lda, ldb);
  }

  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, unsigned int lda,
                     T const *B, unsigned int ldb, float beta, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueDefault, alpha, A, B, beta, C, C, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n},
                  lda, ldb);
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, TA const &A, unsigned int lda,
                   TB const &B, unsigned int ldb, float beta, TBias &bias,
                   TC &C) {
    gemm(transa, transb, m, n, k, alpha,
         static_cast<const float *>(alpaka::getPtrNative(A)), lda,
         static_cast<const float *>(alpaka::getPtrNative(B)), ldb, beta,
         static_cast<float *>(alpaka::getPtrNative(bias)),
         static_cast<float *>(alpaka::getPtrNative(C)));
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, TA const &A,
                       unsigned int lda, TB const &B, unsigned int ldb,
                       float beta, TBias &bias, TC &C) {
    gemmrelu(transa, transb, m, n, k, alpha,
             static_cast<const float *>(alpaka::getPtrNative(A)), lda,
             static_cast<const float *>(alpaka::getPtrNative(B)), ldb, beta,
             static_cast<float *>(alpaka::getPtrNative(bias)),
             static_cast<float *>(alpaka::getPtrNative(C)));
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, TA const &A,
                       unsigned int lda, TB const &B, unsigned int ldb,
                       float beta, TBias &bias, TC &C) {
    gemmgelu(transa, transb, m, n, k, alpha,
             static_cast<const float *>(alpaka::getPtrNative(A)), lda,
             static_cast<const float *>(alpaka::getPtrNative(B)), ldb, beta,
             static_cast<float *>(alpaka::getPtrNative(bias)),
             static_cast<float *>(alpaka::getPtrNative(C)));
  }

  template <typename TA, typename TB, typename TC>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, TA const &A, unsigned int lda,
                     TB const &B, unsigned int ldb, float beta, TC &C) {
    matmul(transa, transb, m, n, k, alpha,
           static_cast<const float *>(alpaka::getPtrNative(A)), lda,
           static_cast<const float *>(alpaka::getPtrNative(B)), ldb, beta,
           static_cast<float *>(alpaka::getPtrNative(C)));
  }

  // Batched multiply with a fused epilogue: for every batch b in
  // [0, batchCount), C_b = epilogue(alpha * op(A_b) * op(B_b) + beta * C_b +
  // bias_b), with X_b = X + b * strideX (strides in elements, column-major
  // matrices with leading dimensions lda, ldb, ldc). A stride of 0 shares the
  // matrix between the batches (broadcast). bias is a vector with one element per
  // row of C (m), bias_b = bias + b * strideBias; Epilogue::Bias adds the bias, 
  // ReluBias and GeluBias also apply the activation.
  inline void gemmStridedBatched(char transa, char transb, int m, int n, int k,
                                 float alpha, const float *A, int lda,
                                 long long strideA, const float *B, int ldb,
                                 long long strideB, float beta, float *C,
                                 int ldc, long long strideC, int batchCount,
                                 Epilogue epilogue, const float *bias,
                                 long long strideBias = 0) {
    if (batchCount < 1)
      throw std::invalid_argument("sofieBLAS: batchCount must be positive.");
    const auto epi = toApiEpilogue(epilogue);
    const auto opA = charToTranspose(transa);
    const auto opB = charToTranspose(transb);
    const auto shapeA = layoutKeyA(transa, m, k);
    const auto shapeB = layoutKeyB(transb, k, n);
    if (static_cast<std::size_t>(lda) < shapeA.first ||
        static_cast<std::size_t>(ldb) < shapeB.first ||
        static_cast<std::size_t>(ldc) < static_cast<std::size_t>(m))
      throw std::invalid_argument(
          "sofieBLAS: leading dimension smaller than the number of rows of "
          "the matrix.");

    BatchKey key{(long long)opA,    (long long)opB,    (long long)epi,
                 (long long)m,      (long long)n,      (long long)k,
                 (long long)lda,    (long long)ldb,    (long long)ldc,
                 strideA,           strideB,           strideC,
                 (long long)batchCount};
    auto it = batchedStore.find(key);
    if (it == batchedStore.end()) {
      BatchedEntry e{};
      e.lA = createBatchedLayout(shapeA, lda, strideA, batchCount);
      e.lB = createBatchedLayout(shapeB, ldb, strideB, batchCount);
      e.lC = createBatchedLayout({(std::size_t)m, (std::size_t)n}, ldc,
                                 strideC, batchCount);
      auto &desc = getOrCreateDesc(opA, opB, epi);
      int returned = 0;
      SOFIEBLAS_CHECK_LT(Api::getHeuristic(ltHandle, desc, e.lA, e.lB, e.lC,
                                           e.lC, preference, 1, &e.h,
                                           &returned));
      if (returned == 0) {
        std::cerr << "[sofieBLAS] No suitable " << Api::name
                  << " algorithm found for the batched multiply m=" << m
                  << " n=" << n << " k=" << k << " batchCount=" << batchCount
                  << " strideA=" << strideA << " strideB=" << strideB << "\n";
        exit(EXIT_FAILURE);
      }
      it = batchedStore.emplace(key, e).first;
    }
    auto &e = it->second;

    auto &desc = getOrCreateDesc(opA, opB, epi);
    if (epi != Api::EpilogueDefault) {
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasPointer,
                                               &bias, sizeof(bias)));
      std::int64_t biasStride = strideBias;
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasBatchStride,
                                               &biasStride,
                                               sizeof(biasStride)));
    }
    SOFIEBLAS_CHECK_LT(Api::matmul(ltHandle, desc, &alpha, A, e.lA, B, e.lB,
                                   &beta, C, e.lC, C, e.lC, &e.h.algo,
                                   d_workspace, workspaceSize, stream));
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
    executeI8Matmul(charToTranspose(transa), charToTranspose(transb),
                    reinterpret_cast<const int8_t *>(alpaka::getPtrNative(A)),
                    reinterpret_cast<const int8_t *>(alpaka::getPtrNative(B)),
                    reinterpret_cast<int32_t *>(alpaka::getPtrNative(C)),
                    layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k,
                         const int8_t *A, const int8_t *B, int32_t *C) {
    executeI8Matmul(charToTranspose(transa), charToTranspose(transb),
                    A, B, C,
                    layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

private:
  typename Api::Queue m_queue;

  static typename Api::Epilogue toApiEpilogue(Epilogue epilogue) {
    switch (epilogue) {
    case Epilogue::Bias:
      return Api::EpilogueBias;
    case Epilogue::ReluBias:
      return Api::EpilogueReluBias;
    case Epilogue::GeluBias:
      return Api::EpilogueGeluBias;
    case Epilogue::Default:
      break;
    }
    return Api::EpilogueDefault;
  }

  // the configuration of a batched multiply: transposes, epilogue, m, n, k,
  // the leading dimensions, the three strides and the batch count
  using BatchKey = std::array<long long, 13>;
  struct BatchedEntry {
    typename Api::Layout lA = nullptr, lB = nullptr, lC = nullptr;
    typename Api::HeuristicResult h{};
  };
  std::map<BatchKey, BatchedEntry> batchedStore;

  typename Api::Layout
  createBatchedLayout(const std::pair<std::size_t, std::size_t> &shape,
                      std::size_t ld, long long stride, int batchCount) {
    typename Api::Layout layout = nullptr;
    SOFIEBLAS_CHECK_LT(Api::layoutCreate(&layout, Api::RealF32, shape.first,
                                         shape.second, ld));
    std::int32_t count = batchCount;
    std::int64_t offset = stride;
    SOFIEBLAS_CHECK_LT(Api::layoutSetAttribute(
        layout, Api::LayoutBatchCount, &count, sizeof(count)));
    SOFIEBLAS_CHECK_LT(Api::layoutSetAttribute(
        layout, Api::LayoutStridedBatchOffset, &offset, sizeof(offset)));
    return layout;
  }

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

  // Returns the layout describing a (rows, cols) matrix with leading dimension
  // ld (column-major), creating and caching it on first use. The dense layouts
  // (ld = rows) are keyed by the shape alone, the others also by ld.
  typename Api::Layout
  getOrCreateLayout(const std::pair<std::size_t, std::size_t> &shape,
                    std::size_t ld) {
    if (ld < shape.first)
      throw std::invalid_argument(
          std::string("sofieBLAS: leading dimension ") + std::to_string(ld) +
          " is smaller than the number of rows " + std::to_string(shape.first) +
          " of the matrix.");
    if (ld == shape.first) {
      auto it = layoutStore.find(shape);
      if (it != layoutStore.end())
        return it->second;
      typename Api::Layout layout = nullptr;
      SOFIEBLAS_CHECK_LT(Api::layoutCreate(&layout, Api::RealF32, shape.first,
                                           shape.second, ld));
      layoutStore.emplace(shape, layout);
      return layout;
    }
    LdKey key{shape.first, shape.second, ld};
    auto it = ldLayoutStore.find(key);
    if (it != ldLayoutStore.end())
      return it->second;
    typename Api::Layout layout = nullptr;
    SOFIEBLAS_CHECK_LT(Api::layoutCreate(&layout, Api::RealF32, shape.first,
                                         shape.second, ld));
    ldLayoutStore.emplace(key, layout);
    return layout;
  }

  typename Api::MatmulDesc &getOrCreateDesc(typename Api::Operation transA,
                                            typename Api::Operation transB,
                                            typename Api::Epilogue epilogue) {
    DescKey key{(int)transA, (int)transB, (int)epilogue};
    auto it = descStore.find(key);
    if (it != descStore.end())
      return it->second;

    typename Api::MatmulDesc desc = nullptr;
    SOFIEBLAS_CHECK_LT(Api::descCreate(&desc, Api::ComputeF32, Api::RealF32));
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
    descStore.emplace(key, desc);
    return descStore.at(key);
  }

  typename Api::HeuristicResult &
  getOrComputeAlgo(typename Api::Operation transA,
                   typename Api::Operation transB,
                   typename Api::Epilogue epilogue,
                   const std::pair<std::size_t, std::size_t> &shapeA,
                   const std::pair<std::size_t, std::size_t> &shapeB,
                   const std::pair<std::size_t, std::size_t> &shapeC,
                   std::size_t ldA = 0, std::size_t ldB = 0) {
    // ldA/ldB are 0 for dense operands
    if (ldA == shapeA.first)
      ldA = 0;
    if (ldB == shapeB.first)
      ldB = 0;
    AlgoKey key{{(int)transA, (int)transB, (int)epilogue},
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
      return it->second.h;
    }

    auto &desc = getOrCreateDesc(transA, transB, epilogue);
    auto lA = getOrCreateLayout(shapeA, ldA ? ldA : shapeA.first);
    auto lB = getOrCreateLayout(shapeB, ldB ? ldB : shapeB.first);
    auto lC = getOrCreateLayout(shapeC, shapeC.first);
    typename Api::HeuristicResult h{};
    int returnedResults = 0;
    SOFIEBLAS_CHECK_LT(Api::getHeuristic(ltHandle, desc, lA, lB, lC, lC,
                                         preference, 1, &h, &returnedResults));
    if (returnedResults == 0) {
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
    return ins->second.h;
  }

  std::unordered_map<std::pair<std::size_t, std::size_t>, typename Api::Layout,
                     PairHash, PairEq>
      i8LayoutStore;

  std::unordered_map<DescKey, typename Api::MatmulDesc, DescKeyHash> i8DescStore;

  struct I8CacheEntry {
    typename Api::HeuristicResult h{};
    std::list<AlgoKey>::iterator lru{};
  };
  std::unordered_map<AlgoKey, I8CacheEntry, AlgoKeyHash> i8AlgoCache;
  std::list<AlgoKey> i8LruOrder;

  typename Api::Layout
  getOrCreateI8Layout(const std::pair<std::size_t, std::size_t> &shape,
                      std::size_t ld, bool isOutput) {
    auto it = i8LayoutStore.find(shape);
    if (it != i8LayoutStore.end())
      return it->second;
    typename Api::Layout layout = nullptr;
    auto dtype = isOutput ? Api::RealI32 : Api::RealI8;
    SOFIEBLAS_CHECK_LT(Api::layoutCreate(&layout, dtype,
                                         shape.first, shape.second, ld));
    i8LayoutStore.emplace(shape, layout);
    return layout;
  }

  typename Api::MatmulDesc &
  getOrCreateI8Desc(typename Api::Operation transA,
                    typename Api::Operation transB) {
    DescKey key{(int)transA, (int)transB, /*epilogue=*/0};
    auto it = i8DescStore.find(key);
    if (it != i8DescStore.end())
      return it->second;

    typename Api::MatmulDesc desc = nullptr;
    SOFIEBLAS_CHECK_LT(Api::descCreate(&desc, Api::ComputeI32, Api::RealI32));
    SOFIEBLAS_CHECK_LT(
        Api::descSetAttribute(desc, Api::DescTransA, &transA, sizeof(transA)));
    SOFIEBLAS_CHECK_LT(
        Api::descSetAttribute(desc, Api::DescTransB, &transB, sizeof(transB)));
    i8DescStore.emplace(key, desc);
    return i8DescStore.at(key);
  }

  typename Api::HeuristicResult &
  getOrComputeI8Algo(typename Api::Operation transA,
                     typename Api::Operation transB,
                     const std::pair<std::size_t, std::size_t> &shapeA,
                     const std::pair<std::size_t, std::size_t> &shapeB,
                     const std::pair<std::size_t, std::size_t> &shapeC) {
    AlgoKey key{{(int)transA, (int)transB, /*epilogue=*/0},
                shapeA.first, shapeA.second,
                shapeB.first, shapeB.second};
    auto it = i8AlgoCache.find(key);
    if (it != i8AlgoCache.end()) {
      if (algoCacheLimit)
        i8LruOrder.splice(i8LruOrder.begin(), i8LruOrder, it->second.lru);
      return it->second.h;
    }

    auto &desc = getOrCreateI8Desc(transA, transB);
    auto lA = getOrCreateI8Layout(shapeA, shapeA.first, /*isOutput=*/false);
    auto lB = getOrCreateI8Layout(shapeB, shapeB.first, /*isOutput=*/false);

    auto outKey = std::make_pair(shapeC.first + (std::size_t(1) << 48),
                                shapeC.second);
    auto lC_it = i8LayoutStore.find(outKey);
    typename Api::Layout lC = nullptr;
    if (lC_it != i8LayoutStore.end()) {
      lC = lC_it->second;
    } else {
      SOFIEBLAS_CHECK_LT(Api::layoutCreate(&lC, Api::RealI32,
                                           shapeC.first, shapeC.second,
                                           shapeC.first));
      i8LayoutStore.emplace(outKey, lC);
    }

    typename Api::HeuristicResult h{};
    int returnedResults = 0;
    SOFIEBLAS_CHECK_LT(Api::getHeuristic(ltHandle, desc, lA, lB, lC, lC,
                                         preference, 1, &h, &returnedResults));
    if (returnedResults == 0) {
      std::cerr << "[sofieBLAS] No suitable " << Api::name
                << " INT8 algorithm found for "
                << "A=[" << shapeA.first << "x" << shapeA.second << "]"
                << " B=[" << shapeB.first << "x" << shapeB.second << "]\n";
      exit(EXIT_FAILURE);
    }
    auto ins = i8AlgoCache.emplace(key, I8CacheEntry{h, {}}).first;
    if (algoCacheLimit) {
      i8LruOrder.push_front(key);
      ins->second.lru = i8LruOrder.begin();
      while (i8AlgoCache.size() > algoCacheLimit) {
        i8AlgoCache.erase(i8LruOrder.back());
        i8LruOrder.pop_back();
      }
    }
    return ins->second.h;
  }

  void executeI8Matmul(typename Api::Operation transA,
                       typename Api::Operation transB,
                       const int8_t *A, const int8_t *B,
                       int32_t *C,
                       const std::pair<std::size_t, std::size_t> &shapeA,
                       const std::pair<std::size_t, std::size_t> &shapeB,
                       const std::pair<std::size_t, std::size_t> &shapeC) {
    auto &h = getOrComputeI8Algo(transA, transB, shapeA, shapeB, shapeC);
    auto &desc = getOrCreateI8Desc(transA, transB);

    auto lA = getOrCreateI8Layout(shapeA, shapeA.first, false);
    auto lB = getOrCreateI8Layout(shapeB, shapeB.first, false);
    auto outKey = std::make_pair(shapeC.first + (std::size_t(1) << 48),
                                shapeC.second);
    auto lC = i8LayoutStore.at(outKey);

    int32_t alpha = 1, beta = 0;
    SOFIEBLAS_CHECK_LT(Api::matmul(ltHandle, desc, &alpha, A, lA, B, lB, &beta,
                                   C, lC, C, lC, &h.algo, d_workspace,
                                   workspaceSize, stream));
  }

    void executeMatmul(typename Api::Operation transA,
                     typename Api::Operation transB,
                     typename Api::Epilogue epilogue, float alpha,
                     const float *A, const float *B, float beta,
                     const float *D_in, float *C_out, const void *bias_ptr,
                     const std::pair<std::size_t, std::size_t> &shapeA,
                     const std::pair<std::size_t, std::size_t> &shapeB,
                     const std::pair<std::size_t, std::size_t> &shapeC,
                     std::size_t ldA = 0, std::size_t ldB = 0) {
    // Retrieve (or lazily compute) the cached algorithm for this shape
    auto &h = getOrComputeAlgo(transA, transB, epilogue, shapeA, shapeB, shapeC,
                               ldA, ldB);

    // Retrieve the cached descriptor and patch the real bias pointer in-place
    auto &desc = getOrCreateDesc(transA, transB, epilogue);
    if (bias_ptr) {
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasPointer,
                                               &bias_ptr, sizeof(bias_ptr)));
    }

    auto lA = getOrCreateLayout(shapeA, ldA ? ldA : shapeA.first);
    auto lB = getOrCreateLayout(shapeB, ldB ? ldB : shapeB.first);
    auto lC = getOrCreateLayout(shapeC, shapeC.first);
    SOFIEBLAS_CHECK_LT(Api::matmul(ltHandle, desc, &alpha, A, lA, B, lB, &beta,
                                   D_in, lC, C_out, lC, &h.algo, d_workspace,
                                   workspaceSize, stream));
  }
};
