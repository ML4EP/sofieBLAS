// The public API of the cuBLASLt and hipBLASLt backends, written once against
// an Api table. A vendor header defines that table (the types, constants and
// functions of its library) and the check macros SOFIEBLAS_CHECK_LT and
// SOFIEBLAS_CHECK_RT, then includes this file.

#include "sofieBLAS/backends/gpu/detail/sofieBLAS_blaslt_utils.hpp"

template <class Api> class BlasLt : private BlasLtBase<Api> {
  using Base = BlasLtBase<Api>;
  using typename Base::BatchKey;
  using typename Base::BatchedEntry;
  using Base::batchedStore;
  using Base::charToTranspose;
  using Base::createBatchedLayout;
  using Base::d_workspace;
  using Base::executeFp8;
  using Base::executeMatmul;
  using Base::executeQuantBatched;
  using Base::floatCaches;
  using Base::fp16Caches;
  using Base::fp8Caches;
  using Base::getOrComputeAlgo;
  using Base::getOrCreateDesc;
  using Base::getOrCreateLayout;
  using Base::handle;
  using Base::int8Caches;
  using Base::layoutKeyA;
  using Base::layoutKeyB;
  using Base::ltHandle;
  using Base::preference;
  using Base::stream;
  using Base::toApiEpilogue;
  using Base::usesBias;
  using Base::workspaceSize;

public:
  // the bf16 and fp8 element types of this backend, for the fp8 GEMMs
  using Bf16 = typename Api::Bf16;
  using Fp8E4M3 = typename Api::Fp8E4M3;
  using Fp8E5M2 = typename Api::Fp8E5M2;

  // the same types under the names shared with the CPU backend
  using Fp8E4M3Type = typename Api::Fp8E4M3;
  using Bf16Type = typename Api::Bf16;

  std::size_t algoCacheSize() const {
    return floatCaches.algoCache.size() + fp16Caches.algoCache.size() +
           int8Caches.algoCache.size() + fp8Caches.algoCache.size();
  }

  BlasLt(const BlasLt &) = delete;
  BlasLt &operator=(const BlasLt &) = delete;
  BlasLt(BlasLt &&) = delete;
  BlasLt &operator=(BlasLt &&) = delete;

  BlasLt(typename Api::Queue &queue, std::size_t cacheLimit = 0)
      : Base(queue, cacheLimit) {}

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

  // float GEMMs: A, B, bias and C all in float. gemm, gemmrelu and gemmgelu
  // add a bias and apply the named activation; matmul has no bias.
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

  // Views only: raw pointers of mixed constness go to the pointer overload.
  template <typename TA, typename TB, typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
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

  // Variants taking the leading dimensions lda and ldb of A and B (column-major,
  // as the physical rows of the matrix: lda >= m for 'N', >= k for 'T'; ldb >= k
  // for 'N', >= n for 'T'). C is dense.
  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, unsigned int lda,
                   T const *B, unsigned int ldb, float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueBias, &alpha, A, B, &beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n}, lda, ldb);
  }

  template <typename T>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A,
                       unsigned int lda, T const *B, unsigned int ldb,
                       float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueReluBias, &alpha, A, B, &beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n}, lda, ldb);
  }

  template <typename T>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A,
                       unsigned int lda, T const *B, unsigned int ldb,
                       float beta, T *bias, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueGeluBias, &alpha, A, B, &beta, bias, C,
                  static_cast<const void *>(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n}, lda, ldb);
  }

  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, unsigned int lda,
                     T const *B, unsigned int ldb, float beta, T *C) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueDefault, &alpha, A, B, &beta, C, C, nullptr,
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

  // Views only: raw pointers of mixed constness go to the pointer overload.
  template <typename TA, typename TB, typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
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
    if constexpr (!(requires { Api::DescBiasBatchStride; })) {
      if (usesBias(epi) && strideBias != 0 && batchCount > 1) {
        for (int b = 0; b < batchCount; ++b)
          gemmStridedBatched(transa, transb, m, n, k, alpha, A + b * strideA,
                             lda, 0, B + b * strideB, ldb, 0, beta,
                             C + b * strideC, ldc, 0, 1, epilogue,
                             bias + b * strideBias, 0);
        return;
      }
    }
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
    if (usesBias(epi)) {
      SOFIEBLAS_CHECK_LT(Api::descSetAttribute(desc, Api::DescBiasPointer,
                                               &bias, sizeof(bias)));
      if constexpr (requires { Api::DescBiasBatchStride; }) {
        std::int64_t biasStride = strideBias;
        SOFIEBLAS_CHECK_LT(Api::descSetAttribute(
            desc, Api::DescBiasBatchStride, &biasStride, sizeof(biasStride)));
      }
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

  // int8 GEMMs: A and B in int8, summed in int32. Without alpha, C is int32
  // and exact; with a float alpha, a float bias and an epilogue, C is int8.
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
                  0, 0, Api::ComputeI32, Api::RealI32, Api::RealI8,
                  Api::RealI32);
  }

  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k,
                         const int8_t *A, const int8_t *B, int32_t *C) {
    const int32_t alpha = 1, beta = 0;
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  Api::EpilogueDefault, &alpha, A, B, &beta, C, C, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n},
                  0, 0, Api::ComputeI32, Api::RealI32, Api::RealI8,
                  Api::RealI32);
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
                  layoutKeyB(transb, k, n), {m, n}, 0, 0, Api::ComputeI32,
                  Api::RealF32, Api::RealI8, Api::RealI8, Api::RealF32);
  }

  inline void int8Matmul(char transa, char transb, unsigned int m,
                         unsigned int n, unsigned int k, float alpha,
                         const int8_t *A, const int8_t *B, const float *bias,
                         int8_t *C, Epilogue epilogue) {
    const float beta = 0.f;
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  toApiEpilogue(epilogue), &alpha, A, B, &beta, C, C, bias,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n},
                  0, 0, Api::ComputeI32, Api::RealF32, Api::RealI8,
                  Api::RealI8, Api::RealF32);
  }

  // int8 GEMM with one alpha per row of C: C[i, j] = epilogue(alphaVector[i]
  // * sum_p op(A)[i, p] * op(B)[p, j] + bias[i]), rounded and saturated to
  // int8 as in the int8-output int8Matmul. alphaVector is a device vector of m
  // floats, beta is zero; bias may be null when the epilogue adds none.
  // cuBLASLt has kernels for it only with A transposed (TN).
  template <typename TAlpha, typename TA, typename TB, typename TBias,
            typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void int8MatmulVectorAlpha(char transa, char transb, unsigned int m,
                                    unsigned int n, unsigned int k,
                                    TAlpha const &alphaVector, TA const &A,
                                    TB const &B, TBias &bias, TC &C,
                                    Epilogue epilogue) {
    int8MatmulVectorAlpha(transa, transb, m, n, k,
                          alpaka::getPtrNative(alphaVector),
                          alpaka::getPtrNative(A), alpaka::getPtrNative(B),
                          alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                          epilogue);
  }

  inline void int8MatmulVectorAlpha(char transa, char transb, unsigned int m,
                                    unsigned int n, unsigned int k,
                                    const float *alphaVector, const int8_t *A,
                                    const int8_t *B, const float *bias,
                                    int8_t *C, Epilogue epilogue) {
    if constexpr (!(requires { Api::PointerModeAlphaDeviceVectorBetaHost; })) {
      throw std::runtime_error("unsupported");
    } else {
      const float beta = 0.f;
      executeMatmul(charToTranspose(transa), charToTranspose(transb),
                    toApiEpilogue(epilogue), alphaVector, A, B, &beta, C, C,
                    bias, layoutKeyA(transa, m, k), layoutKeyB(transb, k, n),
                    {m, n}, 0, 0, Api::ComputeI32, Api::RealF32, Api::RealI8,
                    Api::RealI8, Api::RealF32, nullptr, nullptr, nullptr,
                    /*alphaVector=*/true);
    }
  }

  // Strided-batched int8 GEMMs: for every batch b in [0, batchCount),
  // C_b = op(A_b) * op(B_b) with X_b = X + b * strideX (strides in elements,
  // dense column-major matrices; a stride of 0 shares the matrix between the
  // batches). The first writes the exact int32 sum; the second scales it by
  // alpha, adds the bias (one value per row of C, shared by the batches),
  // applies the epilogue and writes int8 as the int8-output int8Matmul.
  template <typename TA, typename TB, typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void int8MatmulStridedBatched(char transa, char transb,
                                       unsigned int m, unsigned int n,
                                       unsigned int k, TA const &A,
                                       long long strideA, TB const &B,
                                       long long strideB, TC &C,
                                       long long strideC, int batchCount) {
    int8MatmulStridedBatched(transa, transb, m, n, k, alpaka::getPtrNative(A),
                             strideA, alpaka::getPtrNative(B), strideB,
                             alpaka::getPtrNative(C), strideC, batchCount);
  }

  inline void int8MatmulStridedBatched(char transa, char transb,
                                       unsigned int m, unsigned int n,
                                       unsigned int k, const int8_t *A,
                                       long long strideA, const int8_t *B,
                                       long long strideB, int32_t *C,
                                       long long strideC, int batchCount) {
    const int32_t alpha = 1, beta = 0;
    executeQuantBatched(transa, transb, m, n, k, &alpha, A, strideA, B,
                        strideB, &beta, nullptr, C, strideC, batchCount,
                        Epilogue::Default, Api::ComputeI32, Api::RealI32,
                        Api::RealI8, Api::RealI32, Api::RealF32);
  }

  template <typename TA, typename TB, typename TBias, typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void int8MatmulStridedBatched(char transa, char transb,
                                       unsigned int m, unsigned int n,
                                       unsigned int k, float alpha,
                                       TA const &A, long long strideA,
                                       TB const &B, long long strideB,
                                       TBias &bias, TC &C, long long strideC,
                                       int batchCount, Epilogue epilogue) {
    int8MatmulStridedBatched(transa, transb, m, n, k, alpha,
                             alpaka::getPtrNative(A), strideA,
                             alpaka::getPtrNative(B), strideB,
                             alpaka::getPtrNative(bias),
                             alpaka::getPtrNative(C), strideC, batchCount,
                             epilogue);
  }

  inline void int8MatmulStridedBatched(char transa, char transb,
                                       unsigned int m, unsigned int n,
                                       unsigned int k, float alpha,
                                       const int8_t *A, long long strideA,
                                       const int8_t *B, long long strideB,
                                       const float *bias, int8_t *C,
                                       long long strideC, int batchCount,
                                       Epilogue epilogue) {
    const float beta = 0.f;
    executeQuantBatched(transa, transb, m, n, k, &alpha, A, strideA, B,
                        strideB, &beta, bias, C, strideC, batchCount, epilogue,
                        Api::ComputeI32, Api::RealF32, Api::RealI8, Api::RealI8,
                        Api::RealF32);
  }

  // fp16 GEMMs: A and B in half precision (__half), summed and scaled in
  // float. matmulFp16Fp16 writes a half-precision C with a half-precision
  // bias, matmulFp16Fp32 a float C with a float bias.
  template <typename TA, typename TB, typename TBias, typename TC>
  inline void matmulFp16Fp16(char transa, char transb, unsigned int m,
                             unsigned int n, unsigned int k, float alpha,
                             TA const &A, TB const &B, float beta, TBias &bias,
                             TC &C, Epilogue epilogue = Epilogue::Default) {
    auto *c = alpaka::getPtrNative(C);
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  toApiEpilogue(epilogue), &alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), &beta, c, c,
                  alpaka::getPtrNative(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n}, 0, 0, Api::ComputeF32,
                  Api::RealF32, Api::RealF16, Api::RealF16, Api::RealF16);
  }

  inline void matmulFp16Fp16(char transa, char transb, unsigned int m,
                             unsigned int n, unsigned int k, float alpha,
                             const __half *A, const __half *B, float beta,
                             const __half *bias, __half *C,
                             Epilogue epilogue = Epilogue::Default) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  toApiEpilogue(epilogue), &alpha, A, B, &beta, C, C, bias,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n},
                  0, 0, Api::ComputeF32, Api::RealF32, Api::RealF16,
                  Api::RealF16, Api::RealF16);
  }

  template <typename TA, typename TB, typename TBias, typename TC>
  inline void matmulFp16Fp32(char transa, char transb, unsigned int m,
                             unsigned int n, unsigned int k, float alpha,
                             TA const &A, TB const &B, float beta, TBias &bias,
                             TC &C, Epilogue epilogue = Epilogue::Default) {
    auto *c = alpaka::getPtrNative(C);
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  toApiEpilogue(epilogue), &alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(B), &beta, c, c,
                  alpaka::getPtrNative(bias), layoutKeyA(transa, m, k),
                  layoutKeyB(transb, k, n), {m, n}, 0, 0, Api::ComputeF32,
                  Api::RealF32, Api::RealF16, Api::RealF32, Api::RealF32);
  }

  inline void matmulFp16Fp32(char transa, char transb, unsigned int m,
                             unsigned int n, unsigned int k, float alpha,
                             const __half *A, const __half *B, float beta,
                             const float *bias, float *C,
                             Epilogue epilogue = Epilogue::Default) {
    executeMatmul(charToTranspose(transa), charToTranspose(transb),
                  toApiEpilogue(epilogue), &alpha, A, B, &beta, C, C, bias,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n},
                  0, 0, Api::ComputeF32, Api::RealF32, Api::RealF16,
                  Api::RealF32, Api::RealF32);
  }

  // fp8 GEMMs: A and B in FP8 E4M3, each with its own scale (a device pointer
  // to one float), summed and scaled in float, with beta 0. The bias, for the
  // epilogues that add one, is BF16. The output is float, half, bf16, or FP8
  // (E4M3 or E5M2) with a scale of its own. 
  template <typename TA, typename TSA, typename TB, typename TSB,
            typename TBias, typename TC>
  inline void matmulFp8Fp32(char transa, char transb, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            TA const &A, TSA const &scaleA, TB const &B,
                            TSB const &scaleB, TBias &bias, TC &C,
                            Epilogue epilogue = Epilogue::Default) {
    matmulFp8Fp32(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(scaleA), alpaka::getPtrNative(B),
                  alpaka::getPtrNative(scaleB), alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C), epilogue);
  }

  inline void matmulFp8Fp32(char transa, char transb, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            const Fp8E4M3 *A, const float *scaleA,
                            const Fp8E4M3 *B, const float *scaleB,
                            const Bf16 *bias, float *C,
                            Epilogue epilogue = Epilogue::Default) {
    executeFp8(transa, transb, m, n, k, alpha, A, scaleA, B, scaleB, bias, C,
               Api::RealF32, nullptr, epilogue);
  }

  template <typename TA, typename TSA, typename TB, typename TSB,
            typename TBias, typename TC>
  inline void matmulFp8Fp16(char transa, char transb, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            TA const &A, TSA const &scaleA, TB const &B,
                            TSB const &scaleB, TBias &bias, TC &C,
                            Epilogue epilogue = Epilogue::Default) {
    matmulFp8Fp16(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(scaleA), alpaka::getPtrNative(B),
                  alpaka::getPtrNative(scaleB), alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C), epilogue);
  }

  inline void matmulFp8Fp16(char transa, char transb, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            const Fp8E4M3 *A, const float *scaleA,
                            const Fp8E4M3 *B, const float *scaleB,
                            const Bf16 *bias, __half *C,
                            Epilogue epilogue = Epilogue::Default) {
    executeFp8(transa, transb, m, n, k, alpha, A, scaleA, B, scaleB, bias, C,
               Api::RealF16, nullptr, epilogue);
  }

  template <typename TA, typename TSA, typename TB, typename TSB,
            typename TBias, typename TC>
  inline void matmulFp8Bf16(char transa, char transb, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            TA const &A, TSA const &scaleA, TB const &B,
                            TSB const &scaleB, TBias &bias, TC &C,
                            Epilogue epilogue = Epilogue::Default) {
    matmulFp8Bf16(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(scaleA), alpaka::getPtrNative(B),
                  alpaka::getPtrNative(scaleB), alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C), epilogue);
  }

  inline void matmulFp8Bf16(char transa, char transb, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            const Fp8E4M3 *A, const float *scaleA,
                            const Fp8E4M3 *B, const float *scaleB,
                            const Bf16 *bias, Bf16 *C,
                            Epilogue epilogue = Epilogue::Default) {
    executeFp8(transa, transb, m, n, k, alpha, A, scaleA, B, scaleB, bias, C,
               Api::RealBF16, nullptr, epilogue);
  }

  template <typename TA, typename TSA, typename TB, typename TSB,
            typename TBias, typename TC, typename TSD>
  inline void matmulFp8Fp8(char transa, char transb, unsigned int m,
                           unsigned int n, unsigned int k, float alpha,
                           TA const &A, TSA const &scaleA, TB const &B,
                           TSB const &scaleB, TBias &bias, TC &C,
                           TSD const &scaleD,
                           Epilogue epilogue = Epilogue::Default) {
    matmulFp8Fp8(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
                 alpaka::getPtrNative(scaleA), alpaka::getPtrNative(B),
                 alpaka::getPtrNative(scaleB), alpaka::getPtrNative(bias),
                 alpaka::getPtrNative(C), alpaka::getPtrNative(scaleD),
                 epilogue);
  }

  inline void matmulFp8Fp8(char transa, char transb, unsigned int m,
                           unsigned int n, unsigned int k, float alpha,
                           const Fp8E4M3 *A, const float *scaleA,
                           const Fp8E4M3 *B, const float *scaleB,
                           const Bf16 *bias, Fp8E4M3 *C, const float *scaleD,
                           Epilogue epilogue = Epilogue::Default) {
    executeFp8(transa, transb, m, n, k, alpha, A, scaleA, B, scaleB, bias, C,
               Api::RealF8E4M3, scaleD, epilogue);
  }

  template <typename TA, typename TSA, typename TB, typename TSB,
            typename TBias, typename TC, typename TSD>
  inline void matmulFp8E5m2(char transa, char transb, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            TA const &A, TSA const &scaleA, TB const &B,
                            TSB const &scaleB, TBias &bias, TC &C,
                            TSD const &scaleD,
                            Epilogue epilogue = Epilogue::Default) {
    matmulFp8E5m2(transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
                  alpaka::getPtrNative(scaleA), alpaka::getPtrNative(B),
                  alpaka::getPtrNative(scaleB), alpaka::getPtrNative(bias),
                  alpaka::getPtrNative(C), alpaka::getPtrNative(scaleD),
                  epilogue);
  }

  inline void matmulFp8E5m2(char transa, char transb, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            const Fp8E4M3 *A, const float *scaleA,
                            const Fp8E4M3 *B, const float *scaleB,
                            const Bf16 *bias, Fp8E5M2 *C,
                            const float *scaleD,
                            Epilogue epilogue = Epilogue::Default) {
    executeFp8(transa, transb, m, n, k, alpha, A, scaleA, B, scaleB, bias, C,
               Api::RealF8E5M2, scaleD, epilogue);
  }

  // Strided-batched matmulFp8Fp32: for every batch b, C_b = epilogue(alpha *
  // scaleA * scaleB * op(A_b) * op(B_b) + bias) with X_b = X + b * strideX
  // (strides in elements, dense matrices, 0 shares the matrix). The scales and
  // the BF16 bias (one value per row of C) are shared by the batches.
  template <typename TA, typename TSA, typename TB, typename TSB,
            typename TBias, typename TC,
            typename = std::enable_if_t<!std::is_pointer_v<TC>>>
  inline void matmulFp8Fp32StridedBatched(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha, TA const &A, TSA const &scaleA, long long strideA,
      TB const &B, TSB const &scaleB, long long strideB, TBias &bias, TC &C,
      long long strideC, int batchCount,
      Epilogue epilogue = Epilogue::Default) {
    matmulFp8Fp32StridedBatched(
        transa, transb, m, n, k, alpha, alpaka::getPtrNative(A),
        alpaka::getPtrNative(scaleA), strideA, alpaka::getPtrNative(B),
        alpaka::getPtrNative(scaleB), strideB, alpaka::getPtrNative(bias),
        alpaka::getPtrNative(C), strideC, batchCount, epilogue);
  }

  inline void matmulFp8Fp32StridedBatched(
      char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
      float alpha, const Fp8E4M3 *A, const float *scaleA, long long strideA,
      const Fp8E4M3 *B, const float *scaleB, long long strideB,
      const Bf16 *bias, float *C, long long strideC, int batchCount,
      Epilogue epilogue = Epilogue::Default) {
    const float beta = 0.f;
    executeQuantBatched(transa, transb, m, n, k, &alpha, A, strideA, B,
                        strideB, &beta, bias, C, strideC, batchCount, epilogue,
                        Api::ComputeF32, Api::RealF32, Api::RealF8E4M3,
                        Api::RealF32, Api::RealBF16, scaleA, scaleB);
  }
};
