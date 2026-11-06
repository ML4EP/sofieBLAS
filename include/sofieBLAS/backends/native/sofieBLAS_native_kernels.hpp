#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>


// fp8 elements are handled by the type's own conversions, in the format of the
// backend: the caller is responsible for providing data in that format
template <typename TOut, typename TIn>
ALPAKA_FN_HOST_ACC TOut convertElement(TIn const &v) {
  return static_cast<TOut>(v);
}

enum class Activation { None, Relu, Gelu };

template <Activation A>
using ActivationTag = std::integral_constant<Activation, A>;

template <typename TBias> struct EpilogueParams {
  float alpha = 1.f;
  const float *alphaVector = nullptr;
  const float *scaleA = nullptr;
  const float *scaleB = nullptr;
  const float *scaleD = nullptr;
  const TBias *bias = nullptr;
  Activation activation = Activation::None;

  ALPAKA_FN_HOST_ACC float scale() const {
    float s = alpha;
    if (scaleA)
      s *= *scaleA;
    if (scaleB)
      s *= *scaleB;
    return s;
  }
};

template <typename TOut, Activation Act, typename TAcc, typename TSum,
          typename TBias>
ALPAKA_FN_ACC TOut applyEpilogue(TAcc const &acc, TSum sum, std::size_t i,
                                 EpilogueParams<TBias> const &p, float scale) {
  if constexpr (std::is_same_v<TOut, std::int32_t>) {
    return static_cast<std::int32_t>(sum);
  } else {
    float x =
        (p.alphaVector ? p.alphaVector[i] : scale) * static_cast<float>(sum);
    if (p.bias)
      x += static_cast<float>(p.bias[i]);
    if constexpr (Act == Activation::Relu)
      x = x > 0.f ? x : 0.f;
    else if constexpr (Act == Activation::Gelu)
      x *= 0.5f *
           (1.f + alpaka::math::tanh(acc, 0.7978845608028654f *
                                              (x + 0.044715f * x * x * x)));
    if constexpr (std::is_same_v<TOut, std::int8_t>) {
      x = ::rintf(x);
      return static_cast<std::int8_t>(x > 127.f    ? 127.f
                                      : x < -128.f ? -128.f
                                                   : x);
    } else {
      return convertElement<TOut>(p.scaleD ? x * *p.scaleD : x);
    }
  }
}

struct ConvertKernel {
  template <typename TAcc, typename TIn, typename TOut>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, const TIn *in, TOut *out,
                                std::size_t count) const {
    for (auto e : alpaka::uniformElements(acc, count))
      out[e] = convertElement<TOut>(in[e]);
  }
};

// the anonymous union keeps T's (host-only, for fp8) default constructor out of
// the device code
template <typename T, std::size_t N> struct alignas(sizeof(T) * N) Packed {
  union {
    T v[N];
  };
  ALPAKA_FN_HOST_ACC Packed() {}
};

struct EpilogueKernel {
  template <typename TAcc, typename TSum, typename TOut, typename TBias,
            Activation Act>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, const TSum *sum, TOut *out,
                                std::size_t m, std::size_t n, std::size_t cols,
                                long long strideOut,
                                EpilogueParams<TBias> params,
                                ActivationTag<Act>, bool packed) const {
    const float scale = params.scale();
    const std::size_t quads = (m + 3) / 4;
    for (auto col : alpaka::uniformElementsAlongY(acc, cols)) {
      const std::size_t b = col / n;
      const TSum *s = sum + col * m;
      TOut *o = out + static_cast<long long>(b) * strideOut + (col - b * n) * m;
      for (auto q : alpaka::uniformElementsAlongX(acc, quads)) {
        const std::size_t i0 = q * 4;
        Packed<TSum, 4> in;
        if (packed)
          in = *reinterpret_cast<const Packed<TSum, 4> *>(s + i0);
        else
          for (std::size_t t = 0; t < 4; ++t)
            in.v[t] = i0 + t < m ? s[i0 + t] : TSum{};
        Packed<TOut, 4> res;
        for (std::size_t t = 0; t < 4; ++t)
          res.v[t] = applyEpilogue<TOut, Act>(
              acc, in.v[t], i0 + t < m ? i0 + t : m - 1, params, scale);
        if (packed && std::is_same_v<TOut, std::int8_t>)
          *reinterpret_cast<Packed<TOut, 4> *>(o + i0) = res;
        else
          for (std::size_t t = 0; t < 4 && i0 + t < m; ++t)
            o[i0 + t] = res.v[t];
      }
    }
  }
};

struct GemmKernel {
  static constexpr std::size_t tile = 64;
  static constexpr std::size_t depth = 16;
  static constexpr std::size_t threads = 16;
  static constexpr std::size_t perThread = tile / threads;

  template <typename TAcc, typename TSum, typename TA, typename TB,
            typename TOut, typename TBias, Activation Act>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, TSum, const TA *A, const TB *B,
                                TOut *C, std::size_t m, std::size_t n,
                                std::size_t k, bool transA, bool transB,
                                long long strideA, long long strideB,
                                long long strideC, EpilogueParams<TBias> params,
                                ActivationTag<Act>) const {
    auto &tileA = alpaka::declareSharedVar<TSum[depth][tile], __COUNTER__>(acc);
    auto &tileB = alpaka::declareSharedVar<TSum[depth][tile], __COUNTER__>(acc);
    const auto block = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
    const auto thread = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
    const std::size_t batch = block[0];
    const std::size_t j0 = block[1] * tile, i0 = block[2] * tile;
    const std::size_t ty = thread[1], tx = thread[2];
    const std::size_t t = ty * threads + tx;
    const TA *a = A + static_cast<long long>(batch) * strideA;
    const TB *b = B + static_cast<long long>(batch) * strideB;
    const std::size_t lda = transA ? k : m, ldb = transB ? n : k;

    TSum sum[perThread][perThread] = {};
    for (std::size_t k0 = 0; k0 < k; k0 += depth) {
      for (std::size_t l = t; l < depth * tile; l += threads * threads) {
        const std::size_t ii = l % tile, pa = l / tile;
        const std::size_t i = i0 + ii, ka = k0 + pa;
        tileA[pa][ii] =
            i < m && ka < k
                ? convertElement<TSum>(a[transA ? ka + i * lda : i + ka * lda])
                : TSum{};
        const std::size_t pb = l % depth, jj = l / depth;
        const std::size_t j = j0 + jj, kb = k0 + pb;
        tileB[pb][jj] =
            j < n && kb < k
                ? convertElement<TSum>(b[transB ? j + kb * ldb : kb + j * ldb])
                : TSum{};
      }
      alpaka::syncBlockThreads(acc);
      for (std::size_t p = 0; p < depth; ++p) {
        TSum av[perThread], bv[perThread];
        for (std::size_t r = 0; r < perThread; ++r) {
          av[r] = tileA[p][tx + threads * r];
          bv[r] = tileB[p][ty + threads * r];
        }
        for (std::size_t r = 0; r < perThread; ++r)
          for (std::size_t c = 0; c < perThread; ++c)
            sum[r][c] += av[r] * bv[c];
      }
      alpaka::syncBlockThreads(acc);
    }

    const float scale = params.scale();
    TOut *out = C + static_cast<long long>(batch) * strideC;
    for (std::size_t r = 0; r < perThread; ++r)
      for (std::size_t c = 0; c < perThread; ++c) {
        const std::size_t i = i0 + tx + threads * r;
        const std::size_t j = j0 + ty + threads * c;
        if (i < m && j < n)
          out[i + j * m] =
              applyEpilogue<TOut, Act>(acc, sum[r][c], i, params, scale);
      }
  }
};
