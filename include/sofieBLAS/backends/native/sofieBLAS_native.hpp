#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "sofieBLAS/backends/native/sofieBLAS_native_kernels.hpp"
#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>

template <typename TTag, typename TQueue> class FallbackBlas;

// The native kernels. Only FallbackBlas drives them: its methods own the
// scratch buffer and the argument contracts of the helpers below (device
// pointers that match the sizes passed, a scratch pointer used before the
// next scratch call), so those helpers are not public.
template <typename TTag, typename TQueue> class NativeBlas {
public:
  explicit NativeBlas(TQueue &queue) : queue{queue} {}

  template <typename TSum, typename TA, typename TB, typename TOut,
            typename TBias>
  void gemm(char transa, char transb, unsigned int m, unsigned int n,
            unsigned int k, const TA *A, long long strideA, const TB *B,
            long long strideB, TOut *C, long long strideC, int batchCount,
            EpilogueParams<TBias> const &params) {
    if (m == 0 || n == 0 || batchCount < 1)
      return;
    using Dim = alpaka::DimInt<3u>;
    using Acc = alpaka::TagToAcc<TTag, Dim, std::size_t>;
    using Vec = alpaka::Vec<Dim, std::size_t>;
    const std::size_t tile = GemmKernel::tile;
    const Vec blocks{static_cast<std::size_t>(batchCount),
                     (std::size_t{n} + tile - 1) / tile,
                     (std::size_t{m} + tile - 1) / tile};
    const Vec threads{std::size_t{1}, GemmKernel::threads, GemmKernel::threads};
    withActivation(params.activation, [&](auto activation) {
      alpaka::exec<Acc>(
          queue,
          alpaka::WorkDivMembers<Dim, std::size_t>{
              blocks, threads,
              Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}}},
          GemmKernel{}, TSum{}, A, B, C, std::size_t{m}, std::size_t{n},
          std::size_t{k},
          transa == 'T' || transa == 't' || transa == 'C' || transa == 'c',
          transb == 'T' || transb == 't' || transb == 'C' || transb == 'c',
          strideA, strideB, strideC, params, activation);
    });
  }

private:
  friend class FallbackBlas<TTag, TQueue>;

  template <typename TBias>
  static EpilogueParams<TBias>
  epilogueParams(Epilogue epilogue, float alpha, const float *alphaVector,
                 const float *scaleA, const float *scaleB, const float *scaleD,
                 const TBias *bias) {
    const bool addBias = epilogue == Epilogue::Bias ||
                         epilogue == Epilogue::ReluBias ||
                         epilogue == Epilogue::GeluBias;
    EpilogueParams<TBias> p;
    p.alpha = alpha;
    p.alphaVector = alphaVector;
    p.scaleA = scaleA;
    p.scaleB = scaleB;
    p.scaleD = scaleD;
    p.bias = addBias ? bias : nullptr;
    p.activation =
        epilogue == Epilogue::Relu || epilogue == Epilogue::ReluBias
            ? Activation::Relu
        : epilogue == Epilogue::GeluBias || epilogue == Epilogue::Gelu
            ? Activation::Gelu
            : Activation::None;
    return p;
  }

  void *scratch(std::size_t bytes) {
    if (!scratchBuf || alpaka::getExtentProduct(*scratchBuf) < bytes)
      scratchBuf.emplace(
          alpaka::allocAsyncBufIfSupported<std::byte, std::size_t>(queue,
                                                                   bytes));
    return alpaka::getPtrNative(*scratchBuf);
  }

  template <typename TIn, typename TOut>
  void convert(const TIn *in, TOut *out, std::size_t count) {
    if (count == 0)
      return;
    using Dim = alpaka::DimInt<1u>;
    using Acc = alpaka::TagToAcc<TTag, Dim, std::size_t>;
    const std::size_t threads = 256;
    const std::size_t blocks =
        std::min<std::size_t>((count + threads - 1) / threads, 65535);
    alpaka::exec<Acc>(queue,
                      alpaka::WorkDivMembers<Dim, std::size_t>{blocks, threads,
                                                               std::size_t{1}},
                      ConvertKernel{}, in, out, count);
  }

  template <typename TSum, typename TOut, typename TBias>
  void epilogue(const TSum *sum, TOut *C, unsigned int m, unsigned int n,
                int batchCount, long long strideC,
                EpilogueParams<TBias> const &params) {
    const std::size_t cols = std::size_t{n} * batchCount;
    if (m == 0 || cols == 0)
      return;
    const bool packed =
        m % 4 == 0 &&
        (!std::is_same_v<TOut, std::int8_t> ||
         (strideC % 4 == 0 && reinterpret_cast<std::uintptr_t>(C) % 4 == 0));
    using Dim = alpaka::DimInt<2u>;
    using Acc = alpaka::TagToAcc<TTag, Dim, std::size_t>;
    using Vec = alpaka::Vec<Dim, std::size_t>;
    const std::size_t quads = (std::size_t{m} + 3) / 4;
    const std::size_t threadsX = std::min<std::size_t>(
        256, std::max<std::size_t>(32, (quads + 31) / 32 * 32));
    const std::size_t threadsY = 256 / threadsX;
    const Vec blocks{
        std::min<std::size_t>((cols + threadsY - 1) / threadsY, 65535),
        (quads + threadsX - 1) / threadsX};
    withActivation(params.activation, [&](auto activation) {
      alpaka::exec<Acc>(queue,
                        alpaka::WorkDivMembers<Dim, std::size_t>{
                            blocks, Vec{threadsY, threadsX},
                            Vec{std::size_t{1}, std::size_t{1}}},
                        EpilogueKernel{}, sum, C, std::size_t{m},
                        std::size_t{n}, cols, strideC, params, activation,
                        packed);
    });
  }

  template <typename TFn>
  static void withActivation(Activation activation, TFn &&fn) {
    switch (activation) {
    case Activation::Relu:
      fn(ActivationTag<Activation::Relu>{});
      break;
    case Activation::Gelu:
      fn(ActivationTag<Activation::Gelu>{});
      break;
    default:
      fn(ActivationTag<Activation::None>{});
    }
  }

  using ScratchBuf = alpaka::Buf<alpaka::Dev<TQueue>, std::byte,
                                 alpaka::DimInt<1u>, std::size_t>;
  TQueue queue;
  std::optional<ScratchBuf> scratchBuf;
};
