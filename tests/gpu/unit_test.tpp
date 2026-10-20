
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

// fp16 and int8-output GEMMs
template <typename TTag> static void runGpuLowPrecisionTests() {
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

  // copy a host vector into a new device buffer
  auto upload = [&](const auto &v) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    auto h = alpaka::allocBuf<T, Idx>(hostDev, static_cast<Idx>(v.size()));
    std::copy(v.begin(), v.end(), alpaka::getPtrNative(h));
    auto d = alpaka::allocAsyncBuf<T, Idx>(queue, static_cast<Idx>(v.size()));
    alpaka::memcpy(queue, d, h);
    alpaka::wait(queue);
    return d;
  };
  auto download = [&](auto &d, int n) {
    using T = std::remove_pointer_t<decltype(alpaka::getPtrNative(d))>;
    auto h = alpaka::allocBuf<T, Idx>(hostDev, static_cast<Idx>(n));
    alpaka::memcpy(queue, h, d);
    alpaka::wait(queue);
    return std::vector<T>(alpaka::getPtrNative(h), alpaka::getPtrNative(h) + n);
  };

  // ---- fp16: transposes, alpha and beta, epilogues ----
  {
    // multiples of 0.25 below 16: exact in half, and every sum is exact in float
    constexpr int M = 8, N = 4, K = 8;
    std::vector<float> A(M * K), B(K * N), bias(M), C0(M * N);
    fillSeq(A.data(), M * K, -2.f, 0.25f);
    fillSeq(B.data(), K * N, -3.f, 0.25f);
    fillSeq(bias.data(), M, -4.f, 1.f);
    fillSeq(C0.data(), M * N, 1.f, 0.25f);
    auto dA = upload(toHalf(A));
    auto dB = upload(toHalf(B));
    auto dBiasF = upload(bias);
    auto dBiasH = upload(toHalf(bias));

    struct Case {
      char transA, transB;
      float alpha, beta;
      Epilogue epilogue;
      const char *name;
    };
    for (Case c : {Case{'N', 'N', 1.f, 0.f, Epilogue::Default, "NN"},
                   Case{'T', 'N', 1.f, 0.f, Epilogue::Default, "TN"},
                   Case{'N', 'T', 1.f, 0.f, Epilogue::Default, "NT"},
                   Case{'T', 'T', 1.f, 0.f, Epilogue::Default, "TT"},
                   Case{'N', 'N', 0.5f, 1.f, Epilogue::Default,
                        "alpha=0.5 beta=1"},
                   Case{'N', 'N', 1.f, 0.f, Epilogue::ReluBias, "relu bias"},
                   Case{'N', 'N', 1.f, 0.f, Epilogue::Relu, "relu"},
                   Case{'N', 'N', 1.f, 0.f, Epilogue::GeluBias, "gelu bias"}}) {
      std::vector<float> ref = C0; // beta reads C
      refEpilogue(ref.data(), A.data(), B.data(), bias.data(), M, N, K, c.alpha,
                  c.beta, c.transA == 'T', c.transB == 'T', c.epilogue);
      // the GELU epilogue on the GPU is the tanh approximation of the
      // erf-based reference; everything else is exact
      const float tol = c.epilogue == Epilogue::GeluBias ? 2e-3f : 0.f;

      auto dCF = upload(C0);
      blas.matmulFp16Fp32(c.transA, c.transB, M, N, K, c.alpha, dA, dB, c.beta,
                          dBiasF, dCF, c.epilogue);
      checkClose(download(dCF, M * N).data(), ref.data(), M * N,
                 std::string("matmulFp16Fp32 ") + c.name, tol, tol);

      // a half C: the float result rounded to half
      auto dCH = upload(toHalf(C0));
      blas.matmulFp16Fp16(c.transA, c.transB, M, N, K, c.alpha, dA, dB, c.beta,
                          dBiasH, dCH, c.epilogue);
      checkClose(fromHalf(download(dCH, M * N).data(), M * N).data(),
                 fromHalf(toHalf(ref).data(), M * N).data(), M * N,
                 std::string("matmulFp16Fp16 ") + c.name, tol, tol);
    }
  }

  // ---- fp16: the sum is kept in float and rounded to half only at the end ----
  {
    // row 0 of A is 2048, 1, 1, ..., 1 and B is all ones, so the float sum is
    // 2055. Half numbers above 2048 are 2 apart, so a sum kept in half cannot
    // reach 2055; written as half, 2055 rounds to 2056.
    constexpr int M = 8, N = 4, K = 8;
    std::vector<float> A(M * K, 0.f), B(K * N, 1.f), zeros(M * N, 0.f);
    A[0] = 2048.f;
    for (int p = 1; p < K; ++p)
      A[p * M] = 1.f;
    std::vector<float> ref(M * N, 0.f);
    refMatmul(ref.data(), A.data(), B.data(), M, N, K, 1.f, 0.f, false, false);
    auto dA = upload(toHalf(A));
    auto dB = upload(toHalf(B));
    auto dBiasF = upload(std::vector<float>(M, 0.f));
    auto dBiasH = upload(toHalf(std::vector<float>(M, 0.f)));

    auto dCF = upload(zeros);
    blas.matmulFp16Fp32('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBiasF, dCF);
    checkClose(download(dCF, M * N).data(), ref.data(), M * N,
               "matmulFp16Fp32 sum kept in float (2055)", 0.f, 0.f);

    auto dCH = upload(toHalf(zeros));
    blas.matmulFp16Fp16('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBiasH, dCH);
    checkClose(fromHalf(download(dCH, M * N).data(), M * N).data(),
               fromHalf(toHalf(ref).data(), M * N).data(), M * N,
               "matmulFp16Fp16 rounded to half at the end (2056)", 0.f, 0.f);
  }

  // ---- fp16: a shape that is no multiple of 8, through the raw pointers ----
  {
    constexpr int M = 33, N = 17, K = 9;
    std::vector<float> A(M * K), B(K * N), bias(M), zeros(M * N, 0.f);
    for (int i = 0; i < M * K; ++i)
      A[i] = 0.5f * static_cast<float>(i % 13 - 6);
    for (int i = 0; i < K * N; ++i)
      B[i] = 0.25f * static_cast<float>(i % 7 - 3);
    fillSeq(bias.data(), M, -2.f, 0.5f);
    std::vector<float> ref(M * N, 0.f);
    refEpilogue(ref.data(), A.data(), B.data(), bias.data(), M, N, K, 1.f, 0.f,
                false, false, Epilogue::Bias);
    auto dA = upload(toHalf(A));
    auto dB = upload(toHalf(B));
    auto dBiasF = upload(bias);
    auto dBiasH = upload(toHalf(bias));
    auto dCF = upload(zeros);
    auto dCH = upload(toHalf(zeros));

    blas.matmulFp16Fp32('N', 'N', M, N, K, 1.f, alpaka::getPtrNative(dA),
                        alpaka::getPtrNative(dB), 0.f,
                        alpaka::getPtrNative(dBiasF), alpaka::getPtrNative(dCF),
                        Epilogue::Bias);
    checkClose(download(dCF, M * N).data(), ref.data(), M * N,
               "matmulFp16Fp32 33x17x9 raw pointers", 0.f, 0.f);

    blas.matmulFp16Fp16('N', 'N', M, N, K, 1.f, alpaka::getPtrNative(dA),
                        alpaka::getPtrNative(dB), 0.f,
                        alpaka::getPtrNative(dBiasH), alpaka::getPtrNative(dCH),
                        Epilogue::Bias);
    checkClose(fromHalf(download(dCH, M * N).data(), M * N).data(),
               fromHalf(toHalf(ref).data(), M * N).data(), M * N,
               "matmulFp16Fp16 33x17x9 raw pointers", 0.f, 0.f);
  }

  // ---- int8 output: float alpha, float bias, epilogue ----
  {
    // alpha = 1/3 never lands a whole-number sum on x.5, which keeps clear of
    // ties, whose rounding is not tested; alpha = 3 pushes 13 of the 32
    // results past 127 or below -128
    constexpr int M = 8, N = 4, K = 8;
    std::vector<int8_t> A(M * K), B(K * N);
    for (int i = 0; i < M * K; ++i)
      A[i] = static_cast<int8_t>(i * 5 % 17 - 8);
    for (int i = 0; i < K * N; ++i)
      B[i] = static_cast<int8_t>(i * 3 % 13 - 6);
    std::vector<float> bias(M);
    fillSeq(bias.data(), M, -6.f, 2.f);
    auto dA = upload(A);
    auto dB = upload(B);
    auto dBias = upload(bias);
    auto dC = alpaka::allocAsyncBuf<int8_t, Idx>(queue, static_cast<Idx>(M * N));

    struct Case {
      char transA, transB;
      float alpha;
      Epilogue epilogue;
      bool rawPointers;
      const char *name;
    };
    for (Case c :
         {Case{'N', 'N', 1.f / 3.f, Epilogue::Default, false, "NN"},
          Case{'T', 'N', 1.f / 3.f, Epilogue::Default, true, "TN raw pointers"},
          Case{'N', 'N', 1.f / 3.f, Epilogue::ReluBias, false, "relu bias"},
          Case{'N', 'N', 1.f / 3.f, Epilogue::Relu, false, "relu"},
          Case{'N', 'N', 3.f, Epilogue::Default, false, "saturated"}}) {
      const std::string name = std::string("int8Matmul int8 out ") + c.name;
      // the library may have no kernel for a combination; that is reported,
      // not failed
      if (!blas.int8MatmulSupported(c.transA, c.transB, M, N, K, c.epilogue)) {
        std::cout << "  SKIP  " << name << " (no kernel)\n";
        continue;
      }
      std::vector<int8_t> ref(M * N);
      refMatmulI8Out(ref.data(), A.data(), B.data(), bias.data(), M, N, K,
                     c.alpha, c.transA == 'T', c.transB == 'T', c.epilogue);
      if (c.rawPointers)
        blas.int8Matmul(c.transA, c.transB, M, N, K, c.alpha,
                        alpaka::getPtrNative(dA), alpaka::getPtrNative(dB),
                        alpaka::getPtrNative(dBias), alpaka::getPtrNative(dC),
                        c.epilogue);
      else
        blas.int8Matmul(c.transA, c.transB, M, N, K, c.alpha, dA, dB, dBias, dC,
                        c.epilogue);
      const auto got = download(dC, M * N);
      const std::vector<int32_t> got32(got.begin(), got.end());
      const std::vector<int32_t> ref32(ref.begin(), ref.end());
      checkEqual(got32.data(), ref32.data(), M * N, name);
    }
  }
}

