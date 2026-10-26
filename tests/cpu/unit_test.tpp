// CPU backend tests, included from test.cc

static void runCpuTests() {
  std::cout << "\n=== CPU Tests ===\n";

  alpaka::PlatformCpu platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevCpu, alpaka::Blocking> queue{dev};
  sofieBLAS<alpaka::TagCpuSerial> blas(queue);

  constexpr int M = 4, N = 3, K = 5;

  // Allocate host buffers
  auto hA = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * K));
  auto hB = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * N));
  auto hC = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * N));
  auto hBias = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * N));

  float *A = alpaka::getPtrNative(hA);
  float *B = alpaka::getPtrNative(hB);
  float *C = alpaka::getPtrNative(hC);
  float *bias = alpaka::getPtrNative(hBias);

  fillSeq(A, M * K);
  fillSeq(B, K * N, 1.f, 0.5f);
  fillSeq(bias, M * N, 0.1f, 0.1f);

  std::vector<float> ref(M * N);

  // --- matmul NN ---
  fillVal(C, M * N, 0.f);
  blas.matmul('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hC);
  std::copy(C, C + M * N, ref.data());
  std::fill(ref.begin(), ref.end(), 0.f);
  refMatmul(ref.data(), A, B, M, N, K, 1.f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::matmul NN");

  // --- matmul TN  (A^T: K×M physical → M×K logical) ---
  {
    auto hAt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    fillVal(C, M * N, 0.f);
    blas.matmul('T', 'N', M, N, K, 1.f, hAt, hB, 0.f, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), At, B, M, N, K, 1.f, 0.f, true, false);
    checkClose(C, ref.data(), M * N, "cpu::matmul TN");
  }

  // --- matmul NT ---
  {
    auto hBt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(N * K));
    float *Bt = alpaka::getPtrNative(hBt);
    fillSeq(Bt, N * K, 1.f, 0.5f);
    fillVal(C, M * N, 0.f);
    blas.matmul('N', 'T', M, N, K, 1.f, hA, hBt, 0.f, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), A, Bt, M, N, K, 1.f, 0.f, false, true);
    checkClose(C, ref.data(), M * N, "cpu::matmul NT");
  }

  // --- matmul TT ---
  {
    auto hAt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * M));
    auto hBt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(N * K));
    float *At = alpaka::getPtrNative(hAt);
    float *Bt = alpaka::getPtrNative(hBt);
    fillSeq(At, K * M);
    fillSeq(Bt, N * K, 1.f, 0.5f);
    fillVal(C, M * N, 0.f);
    blas.matmul('T', 'T', M, N, K, 1.f, hAt, hBt, 0.f, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), At, Bt, M, N, K, 1.f, 0.f, true, true);
    checkClose(C, ref.data(), M * N, "cpu::matmul TT");
  }

  // --- matmul: alpha scaling ---
  fillVal(C, M * N, 0.f);
  blas.matmul('N', 'N', M, N, K, 2.5f, hA, hB, 0.f, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refMatmul(ref.data(), A, B, M, N, K, 2.5f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::matmul alpha=2.5");

  // --- matmul: beta accumulation ---
  fillSeq(C, M * N, 10.f); // pre-fill C
  blas.matmul('N', 'N', M, N, K, 1.f, hA, hB, 0.5f, hC);
  {
    std::vector<float> C0(M * N);
    fillSeq(C0.data(), M * N, 10.f);
    refMatmul(ref.data(), A, B, M, N, K, 1.f, 0.5f, false, false);
    std::copy(C0.begin(), C0.end(), ref.data());
    refMatmul(ref.data(), A, B, M, N, K, 1.f, 0.5f, false, false);
  }
  checkClose(C, ref.data(), M * N, "cpu::matmul beta=0.5");

  // --- gemm NN (beta=0, no prior accumulation) ---
  fillVal(C, M * N, 0.f);
  fillSeq(bias, M * N, 0.1f, 0.1f);
  blas.gemm('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hBias, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemm(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::gemm NN beta=0");

  // --- gemm NN (beta=1 accumulation) ---
  fillVal(C, M * N, 0.f);
  blas.gemm('N', 'N', M, N, K, 1.f, hA, hB, 1.f, hBias, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemm(ref.data(), A, B, bias, M, N, K, 1.f, 1.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::gemm NN beta=1");

  // --- gemm TN ---
  {
    auto hAt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    fillVal(C, M * N, 0.f);
    blas.gemm('T', 'N', M, N, K, 1.f, hAt, hB, 0.f, hBias, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemm(ref.data(), At, B, bias, M, N, K, 1.f, 0.f, true, false);
    checkClose(C, ref.data(), M * N, "cpu::gemm TN");
  }

  // --- gemmrelu: all-positive matmul result stays unchanged ---
  {
    // A and B with positive values ensure result is positive before bias
    auto hAp = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * K));
    auto hBp = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * N));
    auto hBiasp = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * N));
    float *Ap = alpaka::getPtrNative(hAp);
    float *Bp = alpaka::getPtrNative(hBp);
    float *biasp = alpaka::getPtrNative(hBiasp);
    fillSeq(Ap, M * K, 0.1f, 0.1f);
    fillSeq(Bp, K * N, 0.1f, 0.1f);
    fillVal(biasp, M * N, 0.f);
    fillVal(C, M * N, 0.f);
    blas.gemmrelu('N', 'N', M, N, K, 1.f, hAp, hBp, 0.f, hBiasp, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmRelu(ref.data(), Ap, Bp, biasp, M, N, K, 1.f, 0.f, false, false);
    checkClose(C, ref.data(), M * N, "cpu::gemmrelu all-positive");
  }

  // --- gemmrelu: negative values clamped to zero ---
  {
    // Use alpha=-1 to force negative results
    auto hBiasz = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * N));
    fillVal(alpaka::getPtrNative(hBiasz), M * N, 0.f);
    fillVal(C, M * N, 0.f);
    blas.gemmrelu('N', 'N', M, N, K, -1.f, hA, hB, 0.f, hBiasz, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmRelu(ref.data(), A, B, alpaka::getPtrNative(hBiasz), M, N, K, -1.f,
                0.f, false, false);
    checkClose(C, ref.data(), M * N, "cpu::gemmrelu alpha=-1 (clamped)");
  }

  // --- gemmrelu with bias ---
  fillVal(C, M * N, 0.f);
  fillSeq(bias, M * N, -5.f, 2.f);
  blas.gemmrelu('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hBias, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmRelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::gemmrelu with mixed bias");

  // --- gemmgelu NN ---
  fillVal(C, M * N, 0.f);
  fillVal(bias, M * N, 0.f);
  blas.gemmgelu('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hBias, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmGelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::gemmgelu NN");

  // --- gemmgelu with bias ---
  fillVal(C, M * N, 0.f);
  fillSeq(bias, M * N, -2.f, 0.5f);
  blas.gemmgelu('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hBias, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmGelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::gemmgelu with bias");

  // --- gemmgelu TN ---
  {
    auto hAt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    fillVal(C, M * N, 0.f);
    fillVal(bias, M * N, 0.f);
    blas.gemmgelu('T', 'N', M, N, K, 1.f, hAt, hB, 0.f, hBias, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmGelu(ref.data(), At, B, bias, M, N, K, 1.f, 0.f, true, false);
    checkClose(C, ref.data(), M * N, "cpu::gemmgelu TN");
  }

  // --- edge: zero matrix ---
  {
    auto hZ = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * K));
    fillVal(alpaka::getPtrNative(hZ), M * K, 0.f);
    fillVal(C, M * N, 99.f);
    fillVal(bias, M * N, 0.f);
    blas.matmul('N', 'N', M, N, K, 1.f, hZ, hB, 0.f, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    checkClose(C, ref.data(), M * N, "cpu::matmul zero-A");
  }

  // --- int8 matmul ---
  {
    auto hAi = alpaka::allocBuf<int8_t, Idx>(dev, static_cast<Idx>(M * K));
    auto hBi = alpaka::allocBuf<int8_t, Idx>(dev, static_cast<Idx>(K * N));
    auto hCi = alpaka::allocBuf<int32_t, Idx>(dev, static_cast<Idx>(M * N));
    int8_t *Ai = alpaka::getPtrNative(hAi);
    int8_t *Bi = alpaka::getPtrNative(hBi);
    int32_t *Ci = alpaka::getPtrNative(hCi);
    fillSeqI8(Ai, M * K, 1, 1);
    fillSeqI8(Bi, K * N, -2, 1);

    std::vector<int32_t> refI8(M * N);

    // --- int8 matmul NN ---
    refMatmulI8(refI8.data(), Ai, Bi, M, N, K, false, false);
    blas.int8Matmul('N', 'N', M, N, K, hAi, hBi, hCi);
    checkEqual(Ci, refI8.data(), M * N, "cpu::int8Matmul NN");

    // --- int8 matmul TN ---
    {
      auto hAit = alpaka::allocBuf<int8_t, Idx>(dev, static_cast<Idx>(K * M));
      int8_t *Ait = alpaka::getPtrNative(hAit);
      fillSeqI8(Ait, K * M, 1, 1);
      refMatmulI8(refI8.data(), Ait, Bi, M, N, K, true, false);
      blas.int8Matmul('T', 'N', M, N, K, hAit, hBi, hCi);
      checkEqual(Ci, refI8.data(), M * N, "cpu::int8Matmul TN");
    }

    // --- int8 matmul NT ---
    {
      auto hBit = alpaka::allocBuf<int8_t, Idx>(dev, static_cast<Idx>(N * K));
      int8_t *Bit = alpaka::getPtrNative(hBit);
      fillSeqI8(Bit, N * K, -2, 1);
      refMatmulI8(refI8.data(), Ai, Bit, M, N, K, false, true);
      blas.int8Matmul('N', 'T', M, N, K, hAi, hBit, hCi);
      checkEqual(Ci, refI8.data(), M * N, "cpu::int8Matmul NT");
    }

    // --- int8 matmul TT ---
    {
      auto hAit = alpaka::allocBuf<int8_t, Idx>(dev, static_cast<Idx>(K * M));
      auto hBit = alpaka::allocBuf<int8_t, Idx>(dev, static_cast<Idx>(N * K));
      int8_t *Ait = alpaka::getPtrNative(hAit);
      int8_t *Bit = alpaka::getPtrNative(hBit);
      fillSeqI8(Ait, K * M, 1, 1);
      fillSeqI8(Bit, N * K, -2, 1);
      refMatmulI8(refI8.data(), Ait, Bit, M, N, K, true, true);
      blas.int8Matmul('T', 'T', M, N, K, hAit, hBit, hCi);
      checkEqual(Ci, refI8.data(), M * N, "cpu::int8Matmul TT");
    }

    // --- int8 matmul: zero A ---
    {
      auto hZero = alpaka::allocBuf<int8_t, Idx>(dev, static_cast<Idx>(M * K));
      int8_t *Zero = alpaka::getPtrNative(hZero);
      std::fill(Zero, Zero + M * K, int8_t{0});
      std::fill(refI8.begin(), refI8.end(), 0);
      blas.int8Matmul('N', 'N', M, N, K, hZero, hBi, hCi);
      checkEqual(Ci, refI8.data(), M * N, "cpu::int8Matmul zero-A");
    }
  }

  // --- leading dimensions larger than the number of rows ---
  {
    std::vector<float> Bn(K * N), Bd(M * N);
    fillSeq(Bn.data(), K * N, 1.f, 0.5f);
    for (bool tA : {false, true}) {
      const int rowsA = tA ? K : M, colsA = tA ? M : K;
      const unsigned lda = rowsA + 3, ldb = K + 2;
      std::vector<float> An(M * K);
      fillSeq(An.data(), M * K, 0.3f, 0.2f);
      auto pA = padColMajor(An.data(), rowsA, colsA, lda);
      auto pB = padColMajor(Bn.data(), K, N, ldb);
      const std::string tag = std::string(tA ? "T" : "N") + "N";

      std::fill(ref.begin(), ref.end(), 0.f);
      refMatmul(ref.data(), An.data(), Bn.data(), M, N, K, 1.f, 0.f, tA, false);
      fillVal(C, M * N, 0.f);
      blas.matmul(tA ? 'T' : 'N', 'N', M, N, K, 1.f, pA.data(), lda, pB.data(),
                  ldb, 0.f, C);
      checkClose(C, ref.data(), M * N, "cpu::matmul ld " + tag);

      refGemmRelu(ref.data(), An.data(), Bn.data(), bias, M, N, K, 1.f, 0.f,
                  tA, false);
      blas.gemmrelu(tA ? 'T' : 'N', 'N', M, N, K, 1.f, pA.data(), lda,
                    pB.data(), ldb, 0.f, bias, C);
      checkClose(C, ref.data(), M * N, "cpu::gemmrelu ld " + tag);
    }
  }

  // --- batched multiply with a fused epilogue ---
  {
    constexpr int BATCH = 3, LDA = M + 2;
    for (bool sharedA : {true, false})
      for (Epilogue epi :
           {Epilogue::Bias, Epilogue::ReluBias, Epilogue::GeluBias}) {
        const long long strideA = sharedA ? 0 : LDA * K;
        const long long strideB = sharedA ? K * N : 0;
        const long long strideBias = sharedA ? M : 0;
        std::vector<float> denseA(M * K * (sharedA ? 1 : BATCH));
        fillSeq(denseA.data(), static_cast<int>(denseA.size()), -1.f, 0.3f);
        std::vector<float> padA(LDA * K * (sharedA ? 1 : BATCH), 99.f);
        for (size_t b = 0; b < (sharedA ? 1u : BATCH); ++b)
          for (int p = 0; p < K; ++p)
            for (int i = 0; i < M; ++i)
              padA[b * LDA * K + p * LDA + i] = denseA[b * M * K + p * M + i];
        std::vector<float> Bb(K * N * (sharedA ? BATCH : 1));
        fillSeq(Bb.data(), static_cast<int>(Bb.size()), -2.f, 0.2f);
        std::vector<float> biasv(M * BATCH);
        fillSeq(biasv.data(), static_cast<int>(biasv.size()), -1.5f, 0.4f);
        std::vector<float> got(M * N * BATCH), want(M * N * BATCH);
        refBatchedEpilogue(want.data(), denseA.data(), M, sharedA ? 0 : M * K,
                           Bb.data(), strideB, biasv.data(), strideBias, M, N,
                           K, BATCH, epi);
        blas.gemmStridedBatched('N', 'N', M, N, K, 1.f, padA.data(), LDA,
                                strideA, Bb.data(), K, strideB, 0.f, got.data(),
                                M, M * N, BATCH, epi, biasv.data(), strideBias);
        checkClose(got.data(), want.data(), M * N * BATCH,
                   std::string("cpu::gemmStridedBatched epilogue ") +
                       (sharedA ? "A shared " : "B shared ") +
                       std::to_string(static_cast<int>(epi)));
      }
  }

  // --- edge: identity-like (square, known result) ---
  {
    constexpr int S = 3;
    auto hI = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(S * S));
    auto hX = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(S * S));
    auto hY = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(S * S));
    float *I = alpaka::getPtrNative(hI);
    float *X = alpaka::getPtrNative(hX);
    float *Y = alpaka::getPtrNative(hY);
    fillVal(I, S * S, 0.f);
    for (int i = 0; i < S; ++i)
      I[i * S + i] = 1.f;
    fillSeq(X, S * S);
    fillVal(Y, S * S, 0.f);
    blas.matmul('N', 'N', S, S, S, 1.f, hI, hX, 0.f, hY);
    checkClose(Y, X, S * S, "cpu::matmul identity×X=X");
  }
}

// int8-output, per-row alpha, strided-batched and fp8 GEMMs
static void runCpuQuantizedTests() {
  std::cout << "\n=== CPU quantized tests ===\n";

  alpaka::PlatformCpu platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevCpu, alpaka::Blocking> queue{dev};
  using Blas = sofieBLAS<alpaka::TagCpuSerial>;
  Blas blas(queue);
  using Fp8 = typename Blas::Fp8E4M3Type;
  using Bf16 = typename Blas::Bf16Type;
  static_assert(sizeof(Fp8) == 1 && sizeof(Bf16) == 2);

  // ---- int8 output: float alpha, float bias, epilogue ----
  {
    // alpha = 0.5 lands odd sums on x.5 (ties go to even), alpha = 3
    // saturates
    constexpr int M = 8, N = 4, K = 8;
    std::vector<int8_t> A(M * K), B(K * N);
    fillI8Pattern(A, B);
    std::vector<float> bias(M);
    fillSeq(bias.data(), M, -6.f, 2.f);
    auto hA = alpaka::allocBuf<int8_t, Idx>(dev, static_cast<Idx>(M * K));
    auto hB = alpaka::allocBuf<int8_t, Idx>(dev, static_cast<Idx>(K * N));
    auto hBias = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M));
    auto hC = alpaka::allocBuf<int8_t, Idx>(dev, static_cast<Idx>(M * N));
    std::copy(A.begin(), A.end(), alpaka::getPtrNative(hA));
    std::copy(B.begin(), B.end(), alpaka::getPtrNative(hB));
    std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(hBias));

    struct Case {
      char transA, transB;
      float alpha;
      Epilogue epilogue;
      const char *name;
    };
    for (Case c : {Case{'N', 'N', 1.f / 3.f, Epilogue::Default, "NN"},
                   Case{'T', 'N', 0.5f, Epilogue::Default, "TN ties"},
                   Case{'N', 'T', 0.5f, Epilogue::Bias, "NT bias ties"},
                   Case{'T', 'T', 1.f / 3.f, Epilogue::ReluBias, "TT relu bias"},
                   Case{'N', 'N', 0.5f, Epilogue::Relu, "relu"},
                   Case{'N', 'N', 3.f, Epilogue::Default, "saturated"}}) {
      std::vector<int8_t> ref(M * N);
      refMatmulI8Out(ref.data(), A.data(), B.data(), bias.data(), M, N, K,
                     c.alpha, c.transA == 'T', c.transB == 'T', c.epilogue);
      blas.int8Matmul(c.transA, c.transB, M, N, K, c.alpha, hA, hB, hBias, hC,
                      c.epilogue);
      checkEqualI8(alpaka::getPtrNative(hC), ref.data(), M * N,
                   std::string("cpu::int8Matmul int8 out ") + c.name);
      std::vector<int8_t> raw(M * N);
      blas.int8Matmul(c.transA, c.transB, M, N, K, c.alpha, A.data(),
                      B.data(), bias.data(), raw.data(), c.epilogue);
      checkEqualI8(raw.data(), ref.data(), M * N,
                   std::string("cpu::int8Matmul int8 out raw ") + c.name);
    }

    // ---- per-row alpha ----
    std::vector<float> alphaVec(M);
    for (int i = 0; i < M; ++i)
      alphaVec[i] = (i % 3 == 0) ? 0.5f : 0.125f * static_cast<float>(i + 1);
    auto hAlpha = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M));
    std::copy(alphaVec.begin(), alphaVec.end(), alpaka::getPtrNative(hAlpha));
    for (Epilogue epi : {Epilogue::Default, Epilogue::Bias, Epilogue::ReluBias,
                         Epilogue::Relu}) {
      std::vector<int8_t> ref(M * N);
      refMatmulI8Out(ref.data(), A.data(), B.data(), bias.data(), M, N, K, 0.f,
                     true, false, epi, alphaVec.data());
      blas.int8MatmulVectorAlpha('T', 'N', M, N, K, hAlpha, hA, hB, hBias, hC,
                                 epi);
      checkEqualI8(alpaka::getPtrNative(hC), ref.data(), M * N,
                   "cpu::int8MatmulVectorAlpha epilogue " +
                       std::to_string(static_cast<int>(epi)));
    }
    {
      // no bias epilogue, null bias, raw pointers
      std::vector<int8_t> ref(M * N), got(M * N);
      refMatmulI8Out(ref.data(), A.data(), B.data(), nullptr, M, N, K, 0.f,
                     false, false, Epilogue::Default, alphaVec.data());
      blas.int8MatmulVectorAlpha('N', 'N', M, N, K, alphaVec.data(), A.data(),
                                 B.data(), nullptr, got.data(),
                                 Epilogue::Default);
      checkEqualI8(got.data(), ref.data(), M * N,
                   "cpu::int8MatmulVectorAlpha raw, null bias");
    }
  }

  // ---- strided-batched int8 (int32 and int8 output) ----
  {
    constexpr int M = 8, N = 4, K = 8, BATCH = 3;
    for (bool sharedA : {true, false}) {
      const long long strideA = sharedA ? 0 : M * K;
      const long long strideB = sharedA ? K * N : 0;
      std::vector<int8_t> A(M * K * (sharedA ? 1 : BATCH)),
          B(K * N * (sharedA ? BATCH : 1));
      fillI8Pattern(A, B);
      std::vector<float> bias(M);
      fillSeq(bias.data(), M, -6.f, 2.f);
      const std::string tag = sharedA ? " A shared" : " B shared";

      std::vector<int32_t> got32(M * N * BATCH), ref32(M * N * BATCH);
      std::vector<int8_t> got8(M * N * BATCH), ref8(M * N * BATCH);
      for (int b = 0; b < BATCH; ++b) {
        refMatmulI8(ref32.data() + b * M * N, A.data() + b * strideA,
                    B.data() + b * strideB, M, N, K, true, false);
        refMatmulI8Out(ref8.data() + b * M * N, A.data() + b * strideA,
                       B.data() + b * strideB, bias.data(), M, N, K, 0.5f,
                       true, false, Epilogue::ReluBias);
      }
      blas.int8MatmulStridedBatched('T', 'N', M, N, K, A.data(), strideA,
                                    B.data(), strideB, got32.data(), M * N,
                                    BATCH);
      checkEqual(got32.data(), ref32.data(), M * N * BATCH,
                 "cpu::int8MatmulStridedBatched int32" + tag);
      blas.int8MatmulStridedBatched('T', 'N', M, N, K, 0.5f, A.data(),
                                    strideA, B.data(), strideB, bias.data(),
                                    got8.data(), M * N, BATCH,
                                    Epilogue::ReluBias);
      checkEqualI8(got8.data(), ref8.data(), M * N * BATCH,
                   "cpu::int8MatmulStridedBatched int8 relu bias" + tag);
    }
  }

  // ---- fp8 conversions ----
  {
    bool pass = true;
    for (int b = 0; b < 256; ++b) {
      const Fp8 v{static_cast<uint8_t>(b)};
      const float f = sofieblas::cpu::fp8E4M3ToFloat(v);
      const float r = refFp8ToFloat(static_cast<uint8_t>(b));
      if ((b & 0x7F) == 0x7F) {
        pass &= std::isnan(f) &&
                (sofieblas::cpu::floatToFp8E4M3(f).bits & 0x7F) == 0x7F;
        continue;
      }
      pass &= f == r && sofieblas::cpu::floatToFp8E4M3(f).bits == b;
    }
    struct Enc {
      float x;
      uint8_t bits;
    };
    for (Enc e : {Enc{448.f, 0x7E}, Enc{460.f, 0x7E}, Enc{1e6f, 0x7E},
                  Enc{-1e6f, 0xFE}, Enc{INFINITY, 0x7E},
                  Enc{1.0625f, 0x38},           // tie 1, 1.125 -> 1
                  Enc{1.1875f, 0x3A},           // tie 1.125, 1.25 -> 1.25
                  Enc{std::ldexp(1.f, -10), 0x00}, // tie 0, 2^-9 -> 0
                  Enc{std::ldexp(3.f, -10), 0x02}, // tie 2^-9, 2^-8 -> 2^-8
                  Enc{std::ldexp(15.5f, -10), 0x08}, // rounds up to 2^-6
                  Enc{-0.f, 0x80}}) {
      if (sofieblas::cpu::floatToFp8E4M3(e.x).bits != e.bits) {
        std::cerr << "  fp8 encode " << e.x << " -> "
                  << int(sofieblas::cpu::floatToFp8E4M3(e.x).bits)
                  << " expected " << int(e.bits) << "\n";
        pass = false;
      }
    }
    pass &= sofieblas::cpu::bf16ToFloat(Bf16{bf16Bits(-2.75f)}) == -2.75f;
    if (pass)
      std::cout << "  PASS  cpu::fp8 E4M3 / bf16 conversions\n";
    else {
      std::cerr << "  FAIL [cpu::fp8 E4M3 / bf16 conversions]\n";
      ++gFailures;
    }
  }

  // ---- fp8 GEMMs ----
  {
    constexpr int M = 16, N = 8, K = 32, BATCH = 3;
    const float alpha = 0.5f, scaleA = 0.25f, scaleB = 2.f, scaleD = 4.f;
    auto bytesA = fp8Pattern(M * K * BATCH, 7u);
    auto bytesB = fp8Pattern(K * N * BATCH, 11u);
    std::vector<Fp8> A(bytesA.size()), B(bytesB.size());
    std::memcpy(A.data(), bytesA.data(), bytesA.size());
    std::memcpy(B.data(), bytesB.data(), bytesB.size());
    std::vector<float> bias(M);
    fillSeq(bias.data(), M, -4.f, 0.5f);
    std::vector<Bf16> biasBf16(M);
    for (int i = 0; i < M; ++i)
      biasBf16[i] = Bf16{bf16Bits(bias[i])};

    for (Epilogue epi : {Epilogue::Default, Epilogue::Bias, Epilogue::ReluBias,
                         Epilogue::Relu}) {
      const std::string tag = " epilogue " + std::to_string(static_cast<int>(epi));
      for (bool tA : {true, false}) {
        auto ref = refFp8Gemm(bytesA.data(), bytesB.data(), bias.data(), M, N,
                              K, alpha, scaleA, scaleB, tA, false, epi);
        std::vector<float> got(M * N);
        blas.matmulFp8Fp32(tA ? 'T' : 'N', 'N', M, N, K, alpha, A.data(),
                           &scaleA, B.data(), &scaleB, biasBf16.data(),
                           got.data(), epi);
        checkClose(got.data(), ref.data(), M * N,
                   std::string("cpu::matmulFp8Fp32 ") + (tA ? "TN" : "NN") +
                       tag,
                   1e-5f, 1e-4f);

        // fp8 C: scaleD * result, rounded to E4M3: within half a step
        std::vector<Fp8> got8(M * N);
        blas.matmulFp8Fp8(tA ? 'T' : 'N', 'N', M, N, K, alpha, A.data(),
                          &scaleA, B.data(), &scaleB, biasBf16.data(),
                          got8.data(), &scaleD, epi);
        std::vector<float> dec(M * N), want(M * N);
        for (int i = 0; i < M * N; ++i) {
          dec[i] = refFp8ToFloat(got8[i].bits);
          want[i] = std::max(-448.f, std::min(448.f, scaleD * ref[i]));
        }
        checkClose(dec.data(), want.data(), M * N,
                   std::string("cpu::matmulFp8Fp8 ") + (tA ? "TN" : "NN") + tag,
                   1.f / 16.f, std::ldexp(1.f, -10));
      }
    }

    // strided-batched, A shared, bias shared
    std::vector<float> want(M * N * BATCH), got(M * N * BATCH);
    for (int b = 0; b < BATCH; ++b) {
      auto ref = refFp8Gemm(bytesA.data(), bytesB.data() + b * K * N,
                            bias.data(), M, N, K, alpha, scaleA, scaleB, true,
                            false, Epilogue::ReluBias);
      std::copy(ref.begin(), ref.end(), want.begin() + b * M * N);
    }
    blas.matmulFp8Fp32StridedBatched('T', 'N', M, N, K, alpha, A.data(),
                                     &scaleA, 0, B.data(), &scaleB, K * N,
                                     biasBf16.data(), got.data(), M * N, BATCH,
                                     Epilogue::ReluBias);
    checkClose(got.data(), want.data(), M * N * BATCH,
               "cpu::matmulFp8Fp32StridedBatched A shared, relu bias", 1e-5f,
               1e-4f);
  }
}
