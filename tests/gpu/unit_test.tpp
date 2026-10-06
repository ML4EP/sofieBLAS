
// shared test for CUDA/HIP

template <typename TTag> static void runGpuTests() {
  std::cout << "\n=== " << __PRETTY_FUNCTION__ << " ===\n";

  using Acc = alpaka::TagToAcc<TTag, Dim1D, Idx>;
  using DevAcc = alpaka::Dev<Acc>;
  using PlatformAcc = alpaka::Platform<Acc>;

  PlatformAcc platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<DevAcc, alpaka::NonBlocking> queue{dev};
  sofieBLAS<TTag> blas(queue);

  alpaka::PlatformCpu hostPlatform{};
  auto hostDev = alpaka::getDevByIdx(hostPlatform, 0u);

  constexpr int M = 4, N = 3, K = 5;

  auto hA = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
  auto hB = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
  auto hC = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));
  auto hBias = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));

  float *A = alpaka::getPtrNative(hA);
  float *B = alpaka::getPtrNative(hB);
  float *bias = alpaka::getPtrNative(hBias);

  fillSeq(A, M * K);
  fillSeq(B, K * N, 1.f, 0.5f);
  fillVal(bias, M * N, 0.f);

  auto dA = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
  auto dB = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
  auto dC = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
  auto dBias =
      alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));

  alpaka::memcpy(queue, dA, hA);
  alpaka::memcpy(queue, dB, hB);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);

  std::vector<float> ref(M * N);
  float *C = alpaka::getPtrNative(hC);

  auto verify = [&](const std::string &name) {
    alpaka::memcpy(queue, hC, dC);
    alpaka::wait(queue);
    checkClose(C, ref.data(), M * N, name);
  };

  // ---- matmul NN ----
  blas.addOperationConfig(M, N, K, ldaFor('N', M, K), ldbFor('N', K, N), M, 'N',
                          'N', Epilogue::Default);
  std::fill(ref.begin(), ref.end(), 0.f);
  refMatmul(ref.data(), A, B, M, N, K, 1.f, 0.f, false, false);
  blas.matmul('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dC);
  verify("matmul NN");

  // ---- matmul TN ----
  {
    auto hAt = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    auto dAt =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * M));
    alpaka::memcpy(queue, dAt, hAt);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, ldaFor('T', M, K), ldbFor('N', K, N), M,
                            'T', 'N', Epilogue::Default);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), At, B, M, N, K, 1.f, 0.f, true, false);
    blas.matmul('T', 'N', M, N, K, 1.f, dAt, dB, 0.f, dC);
    verify("matmul TN");
  }

  // ---- matmul NT ----
  {
    auto hBt = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(N * K));
    float *Bt = alpaka::getPtrNative(hBt);
    fillSeq(Bt, N * K, 1.f, 0.5f);
    auto dBt =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(N * K));
    alpaka::memcpy(queue, dBt, hBt);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, ldaFor('N', M, K), ldbFor('T', K, N), M,
                            'N', 'T', Epilogue::Default);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), A, Bt, M, N, K, 1.f, 0.f, false, true);
    blas.matmul('N', 'T', M, N, K, 1.f, dA, dBt, 0.f, dC);
    verify("matmul NT");
  }

  // ---- matmul alpha=2.5 ----
  std::fill(ref.begin(), ref.end(), 0.f);
  refMatmul(ref.data(), A, B, M, N, K, 2.5f, 0.f, false, false);
  blas.matmul('N', 'N', M, N, K, 2.5f, dA, dB, 0.f, dC);
  verify("matmul alpha=2.5");

  // ---- gemm NN beta=0 ----
  fillSeq(bias, M * N, 0.1f, 0.1f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemm(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemm('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("gemm NN beta=0");

  // ---- gemm NN beta=1 ----
  // D_in = bias, so result = A*B + 1*bias_matrix + bias_vec
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemm(ref.data(), A, B, bias, M, N, K, 1.f, 1.f, false, false);
  blas.gemm('N', 'N', M, N, K, 1.f, dA, dB, 1.f, dBias, dC);
  verify("gemm NN beta=1");

  // ---- gemm TN ----
  {
    auto hAt = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    auto dAt =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * M));
    alpaka::memcpy(queue, dAt, hAt);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, ldaFor('T', M, K), ldbFor('N', K, N), M,
                            'T', 'N', Epilogue::Bias);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemm(ref.data(), At, B, bias, M, N, K, 1.f, 0.f, true, false);
    blas.gemm('T', 'N', M, N, K, 1.f, dAt, dB, 0.f, dBias, dC);
    verify("gemm TN");
  }

  // ---- gemmrelu: all-positive (relu is identity) ----
  {
    auto hAp = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
    auto hBp = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
    auto hBiasz =
        alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));
    float *Ap = alpaka::getPtrNative(hAp);
    float *Bp = alpaka::getPtrNative(hBp);
    fillSeq(Ap, M * K, 0.1f, 0.1f);
    fillSeq(Bp, K * N, 0.1f, 0.1f);
    fillVal(alpaka::getPtrNative(hBiasz), M * N, 0.f);
    auto dAp =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
    auto dBp =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
    auto dBiasz =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
    alpaka::memcpy(queue, dAp, hAp);
    alpaka::memcpy(queue, dBp, hBp);
    alpaka::memcpy(queue, dBiasz, hBiasz);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, M, K, M, 'N', 'N', Epilogue::ReluBias);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmRelu(ref.data(), Ap, Bp, alpaka::getPtrNative(hBiasz), M, N, K, 1.f,
                0.f, false, false);
    blas.gemmrelu('N', 'N', M, N, K, 1.f, dAp, dBp, 0.f, dBiasz, dC);
    verify("gemmrelu all-positive");
  }

  // ---- gemmrelu: alpha=-1 forces negatives -> clamped to zero ----
  {
    auto hBiasz =
        alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));
    fillVal(alpaka::getPtrNative(hBiasz), M * N, 0.f);
    auto dBiasz =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
    alpaka::memcpy(queue, dBiasz, hBiasz);
    alpaka::wait(queue);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmRelu(ref.data(), A, B, alpaka::getPtrNative(hBiasz), M, N, K, -1.f,
                0.f, false, false);
    blas.gemmrelu('N', 'N', M, N, K, -1.f, dA, dB, 0.f, dBiasz, dC);
    verify("gemmrelu alpha=-1 (clamped)");
  }

  // ---- gemmrelu with mixed bias ----
  fillSeq(bias, M * N, -5.f, 2.f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmRelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemmrelu('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("gemmrelu with mixed bias");

  // ---- gemmgelu NN ----
  fillVal(bias, M * N, 0.f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmGelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemmgelu('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("gemmgelu NN");

  // ---- gemmgelu with bias ----
  fillSeq(bias, M * N, -2.f, 0.5f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmGelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemmgelu('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("gemmgelu with bias");

  // ---- edge: zero A ----
  {
    auto hZero = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
    fillVal(alpaka::getPtrNative(hZero), M * K, 0.f);
    auto dZero =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
    alpaka::memcpy(queue, dZero, hZero);
    alpaka::wait(queue);
    std::fill(ref.begin(), ref.end(), 0.f);
    blas.matmul('N', 'N', M, N, K, 1.f, dZero, dB, 0.f, dC);
    verify("matmul zero-A");
  }

  // ---- leading dimensions larger than the number of rows ----
  {
    // A and B are stored in buffers with padded columns (lda = M + 3, ldb =
    // K + 2); the transposed A has K rows and so ld = K + 1
    fillSeq(bias, M * N, -3.f, 1.f);
    alpaka::memcpy(queue, dBias, hBias);
    alpaka::wait(queue);

    struct Case {
      char transA;
      unsigned lda, ldb;
    };
    for (Case c : {Case{'N', M + 3, K + 2}, Case{'T', K + 1, K + 2},
                   Case{'N', M, K + 2}}) {
      const bool tA = c.transA == 'T';
      const int rowsA = tA ? K : M;
      const int colsA = tA ? M : K;
      std::vector<float> At(M * K);
      fillSeq(At.data(), M * K, 0.3f, 0.2f);
      auto pA = padColMajor(At.data(), rowsA, colsA, c.lda);
      auto pB = padColMajor(B, K, N, c.ldb);
      auto hPA = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(pA.size()));
      auto hPB = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(pB.size()));
      std::copy(pA.begin(), pA.end(), alpaka::getPtrNative(hPA));
      std::copy(pB.begin(), pB.end(), alpaka::getPtrNative(hPB));
      auto dPA = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(pA.size()));
      auto dPB = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(pB.size()));
      alpaka::memcpy(queue, dPA, hPA);
      alpaka::memcpy(queue, dPB, hPB);
      alpaka::wait(queue);
      const std::string tag =
          std::string(1, c.transA) + "N lda=" + std::to_string(c.lda) +
          " ldb=" + std::to_string(c.ldb);

      std::fill(ref.begin(), ref.end(), 0.f);
      refMatmul(ref.data(), At.data(), B, M, N, K, 1.f, 0.f, tA, false);
      blas.matmul(c.transA, 'N', M, N, K, 1.f, dPA, c.lda, dPB, c.ldb, 0.f, dC);
      verify("matmul " + tag);

      refGemm(ref.data(), At.data(), B, bias, M, N, K, 1.f, 0.f, tA, false);
      blas.gemm(c.transA, 'N', M, N, K, 1.f, dPA, c.lda, dPB, c.ldb, 0.f,
                dBias, dC);
      verify("gemm " + tag);

      refGemmRelu(ref.data(), At.data(), B, bias, M, N, K, 1.f, 0.f, tA, false);
      blas.gemmrelu(c.transA, 'N', M, N, K, 1.f, dPA, c.lda, dPB, c.ldb, 0.f,
                    dBias, dC);
      verify("gemmrelu " + tag);

      refGemmGelu(ref.data(), At.data(), B, bias, M, N, K, 1.f, 0.f, tA, false);
      blas.gemmgelu(c.transA, 'N', M, N, K, 1.f, dPA, c.lda, dPB, c.ldb, 0.f,
                    dBias, dC);
      verify("gemmgelu " + tag);
    }

    // a leading dimension smaller than the rows is rejected
    bool threw = false;
    try {
      blas.matmul('N', 'N', M, N, K, 1.f, dA, static_cast<unsigned>(M - 1), dB,
                  static_cast<unsigned>(K), 0.f, dC);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    if (threw)
      std::cout << "  PASS  matmul lda < rows throws\n";
    else {
      std::cerr << "  FAIL [matmul lda < rows throws]\n";
      ++gFailures;
    }
  }

  // ---- batched multiply with a fused epilogue ----
  {
    // A (M x K, lda = M + 2) is shared by the batches or B (K x N) is, the
    // bias is per batch or shared
    constexpr int BATCH = 3, LDA = M + 2;
    struct Case {
      bool sharedA;
      long long strideBias;
      Epilogue epilogue;
      const char *name;
    };
    for (Case c : {Case{true, M, Epilogue::Bias, "A shared, bias per batch"},
                   Case{true, M, Epilogue::ReluBias, "A shared, relu"},
                   Case{true, 0, Epilogue::GeluBias, "A shared, gelu"},
                   Case{false, 0, Epilogue::ReluBias, "B shared, relu"},
                   Case{false, M, Epilogue::Bias, "B shared, bias per batch"}}) {
      const long long strideA = c.sharedA ? 0 : LDA * K;
      const long long strideB = c.sharedA ? K * N : 0;
      std::vector<float> hostA(LDA * K * (c.sharedA ? 1 : BATCH), 99.f);
      std::vector<float> denseA(M * K * (c.sharedA ? 1 : BATCH));
      fillSeq(denseA.data(), static_cast<int>(denseA.size()), -1.f, 0.3f);
      for (size_t b = 0; b < (c.sharedA ? 1u : BATCH); ++b)
        for (int p = 0; p < K; ++p)
          for (int i = 0; i < M; ++i)
            hostA[b * LDA * K + p * LDA + i] = denseA[b * M * K + p * M + i];
      std::vector<float> hostB(K * N * (c.sharedA ? BATCH : 1));
      fillSeq(hostB.data(), static_cast<int>(hostB.size()), -2.f, 0.2f);
      std::vector<float> hostBias(M * BATCH);
      fillSeq(hostBias.data(), static_cast<int>(hostBias.size()), -1.5f, 0.4f);

      auto upload = [&](const std::vector<float> &v) {
        auto h = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(v.size()));
        std::copy(v.begin(), v.end(), alpaka::getPtrNative(h));
        auto d = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(v.size()));
        alpaka::memcpy(queue, d, h);
        alpaka::wait(queue);
        return d;
      };
      auto dBA = upload(hostA);
      auto dBB = upload(hostB);
      auto dBBias = upload(hostBias);
      auto dBC = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N * BATCH));

      std::vector<float> refB(M * N * BATCH);
      refBatchedEpilogue(refB.data(), denseA.data(), M, strideA ? M * K : 0,
                         hostB.data(), strideB, hostBias.data(), c.strideBias,
                         M, N, K, BATCH, c.epilogue);
      // the reference reads the dense A, the call the padded one
      blas.gemmStridedBatched('N', 'N', M, N, K, 1.f,
                              alpaka::getPtrNative(dBA), LDA, strideA,
                              alpaka::getPtrNative(dBB), K, strideB, 0.f,
                              alpaka::getPtrNative(dBC), M, M * N, BATCH,
                              c.epilogue, alpaka::getPtrNative(dBBias),
                              c.strideBias);
      auto hBC = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N * BATCH));
      alpaka::memcpy(queue, hBC, dBC);
      alpaka::wait(queue);
      // the GELU epilogue of cuBLASLt/hipBLASLt is the tanh approximation of
      // the erf-based reference, which differs by up to ~1e-3
      const float atol = c.epilogue == Epilogue::GeluBias ? 2e-3f : 1e-4f;
      checkClose(alpaka::getPtrNative(hBC), refB.data(), M * N * BATCH,
                 std::string("gemmStridedBatched epilogue: ") + c.name, 1e-4f,
                 atol);
    }
  }

  // ---- int8 matmul ----
  {
    constexpr int MI = 8, NI = 4, KI = 8;

    auto hAi =
        alpaka::allocBuf<int8_t, Idx>(hostDev, static_cast<Idx>(MI * KI));
    auto hBi =
        alpaka::allocBuf<int8_t, Idx>(hostDev, static_cast<Idx>(KI * NI));
    auto hCi =
        alpaka::allocBuf<int32_t, Idx>(hostDev, static_cast<Idx>(MI * NI));
    int8_t *Ai = alpaka::getPtrNative(hAi);
    int8_t *Bi = alpaka::getPtrNative(hBi);
    int32_t *Ci = alpaka::getPtrNative(hCi);
    fillSeqI8(Ai, MI * KI, 1, 1);
    fillSeqI8(Bi, KI * NI, -2, 1);

    auto dAi =
        alpaka::allocAsyncBuf<int8_t, Idx>(queue, static_cast<Idx>(MI * KI));
    auto dBi =
        alpaka::allocAsyncBuf<int8_t, Idx>(queue, static_cast<Idx>(KI * NI));
    auto dCi =
        alpaka::allocAsyncBuf<int32_t, Idx>(queue, static_cast<Idx>(MI * NI));
    alpaka::memcpy(queue, dAi, hAi);
    alpaka::memcpy(queue, dBi, hBi);
    alpaka::wait(queue);

    std::vector<int32_t> refI8(MI * NI);

    auto verifyI8 = [&](const std::string &name) {
      alpaka::memcpy(queue, hCi, dCi);
      alpaka::wait(queue);
      checkEqual(Ci, refI8.data(), MI * NI, name);
    };

    // ---- int8 matmul NN ----
    refMatmulI8(refI8.data(), Ai, Bi, MI, NI, KI, false, false);
    blas.int8Matmul('N', 'N', MI, NI, KI, dAi, dBi, dCi);
    verifyI8("int8Matmul NN");

    // ---- int8 matmul TN ----
    {
      auto hAit =
          alpaka::allocBuf<int8_t, Idx>(hostDev, static_cast<Idx>(KI * MI));
      int8_t *Ait = alpaka::getPtrNative(hAit);
      fillSeqI8(Ait, KI * MI, 1, 1);
      auto dAit =
          alpaka::allocAsyncBuf<int8_t, Idx>(queue, static_cast<Idx>(KI * MI));
      alpaka::memcpy(queue, dAit, hAit);
      alpaka::wait(queue);
      refMatmulI8(refI8.data(), Ait, Bi, MI, NI, KI, true, false);
      blas.int8Matmul('T', 'N', MI, NI, KI, dAit, dBi, dCi);
      verifyI8("int8Matmul TN");
    }

    // ---- int8 matmul: zero A ----
    {
      auto hZero =
          alpaka::allocBuf<int8_t, Idx>(hostDev, static_cast<Idx>(MI * KI));
      int8_t *Zero = alpaka::getPtrNative(hZero);
      std::fill(Zero, Zero + MI * KI, int8_t{0});
      auto dZero = alpaka::allocAsyncBuf<int8_t, Idx>(
          queue, static_cast<Idx>(MI * KI));
      alpaka::memcpy(queue, dZero, hZero);
      alpaka::wait(queue);
      std::fill(refI8.begin(), refI8.end(), 0);
      blas.int8Matmul('N', 'N', MI, NI, KI, dZero, dBi, dCi);
      verifyI8("int8Matmul zero-A");
    }
  }
}

template <typename TTag>
static void runGpuDynamicShapeTests() {
  std::cout << "\n=== " << __PRETTY_FUNCTION__ << " ===\n";

  using Acc = alpaka::TagToAcc<TTag, Dim1D, Idx>;
  using DevAcc = alpaka::Dev<Acc>;
  using PlatformAcc = alpaka::Platform<Acc>;

  PlatformAcc platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<DevAcc, alpaka::NonBlocking> queue{dev};

  alpaka::PlatformCpu hostPlatform{};
  auto hostDev = alpaka::getDevByIdx(hostPlatform, 0u);

  // M0 is the construction-time size given to addOperationConfig; the buffers
  // hold MCAP rows so sizes above M0 are exercised too.
  constexpr int MCAP = 96, M0 = 64, N = 3, K = 5;

  auto hA = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(MCAP * K));
  auto hB = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
  auto hC = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(MCAP * N));
  float *A = alpaka::getPtrNative(hA);
  float *B = alpaka::getPtrNative(hB);
  float *C = alpaka::getPtrNative(hC);
  fillSeq(A, MCAP * K, 0.5f, 0.25f);
  fillSeq(B, K * N, 1.f, 0.5f);

  auto dA =
      alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(MCAP * K));
  auto dB = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
  auto dC =
      alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(MCAP * N));
  alpaka::memcpy(queue, dA, hA);
  alpaka::memcpy(queue, dB, hB);
  alpaka::wait(queue);

  // One instance serving sizes never passed to addOperationConfig (issue #10),
  // including m=1 and a size above the construction-time one.
  sofieBLAS<TTag> blas(queue);
  blas.addOperationConfig(M0, N, K, ldaFor('N', M0, K), ldbFor('N', K, N), M0,
                          'N', 'N', Epilogue::Default);

  std::vector<float> ref;
  auto runAt = [&](int m, const std::string &name) {
    ref.assign(static_cast<std::size_t>(m) * N, 0.f);
    refMatmul(ref.data(), A, B, m, N, K, 1.f, 0.f, false, false);
    blas.matmul('N', 'N', static_cast<unsigned>(m), static_cast<unsigned>(N),
                static_cast<unsigned>(K), 1.f, dA, dB, 0.f, dC);
    alpaka::memcpy(queue, hC, dC);
    alpaka::wait(queue);
    checkClose(C, ref.data(), m * N, name);
  };

  for (int m : {M0, 37, 8, 51, 1, M0, MCAP})
    runAt(m, "dynamic m=" + std::to_string(m));

  // Generated code calls the raw-pointer overloads; one call keeps them
  // compiled and resolving to the right overload.
  ref.assign(static_cast<std::size_t>(45) * N, 0.f);
  refMatmul(ref.data(), A, B, 45, N, K, 1.f, 0.f, false, false);
  blas.matmul('N', 'N', 45u, static_cast<unsigned>(N), static_cast<unsigned>(K),
              1.f, alpaka::getPtrNative(dA), alpaka::getPtrNative(dB), 0.f,
              alpaka::getPtrNative(dC));
  alpaka::memcpy(queue, hC, dC);
  alpaka::wait(queue);
  checkClose(C, ref.data(), 45 * N, "dynamic raw pointers m=45");

  // 32 distinct sizes through a cache limited to 8 entries.
  {
    sofieBLAS<TTag> capped(queue, 8);
    capped.addOperationConfig(M0, N, K, ldaFor('N', M0, K), ldbFor('N', K, N),
                              M0, 'N', 'N', Epilogue::Default);
    float worst = 0.f;
    for (int m = M0 + 1; m <= MCAP; ++m) {
      ref.assign(static_cast<std::size_t>(m) * N, 0.f);
      refMatmul(ref.data(), A, B, m, N, K, 1.f, 0.f, false, false);
      capped.matmul('N', 'N', static_cast<unsigned>(m),
                    static_cast<unsigned>(N), static_cast<unsigned>(K), 1.f, dA,
                    dB, 0.f, dC);
      alpaka::memcpy(queue, hC, dC);
      alpaka::wait(queue);
      for (std::size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, std::abs(C[i] - ref[i]));
    }
    if (capped.algoCacheSize() <= 8 && worst < 1e-3f) {
      std::cout << "  PASS  cache limit honoured\n";
    } else {
      std::cerr << "  FAIL [cache limit honoured] "
                << capped.algoCacheSize() << " entries, worst err " << worst
                << "\n";
      ++gFailures;
    }
  }
}

