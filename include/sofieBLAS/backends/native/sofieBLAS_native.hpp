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
    using Dimension = alpaka::DimInt<3u>;
    using Accelerator = alpaka::TagToAcc<TTag, Dimension, std::size_t>;
    using Vector = alpaka::Vec<Dimension, std::size_t>;
    const std::size_t tile = GemmKernel::tile;
    const Vector blocks{static_cast<std::size_t>(batchCount),
                        (std::size_t{n} + tile - 1) / tile,
                        (std::size_t{m} + tile - 1) / tile};
    const Vector threads{std::size_t{1}, GemmKernel::threads,
                         GemmKernel::threads};
    withActivation(params.activation, [&](auto activation) {
      alpaka::exec<Accelerator>(
          queue,
          alpaka::WorkDivMembers<Dimension, std::size_t>{
              blocks, threads,
              Vector{std::size_t{1}, std::size_t{1}, std::size_t{1}}},
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
    EpilogueParams<TBias> params;
    params.alpha = alpha;
    params.alphaVector = alphaVector;
    params.scaleA = scaleA;
    params.scaleB = scaleB;
    params.scaleD = scaleD;
    params.bias = addBias ? bias : nullptr;
    params.activation =
        epilogue == Epilogue::Relu || epilogue == Epilogue::ReluBias
            ? Activation::Relu
        : epilogue == Epilogue::GeluBias || epilogue == Epilogue::Gelu
            ? Activation::Gelu
            : Activation::None;
    return params;
  }

  void *scratch(std::size_t bytes) {
    if (!scratchBuf || alpaka::getExtentProduct(*scratchBuf) < bytes)
      scratchBuf.emplace(
          alpaka::allocAsyncBufIfSupported<std::byte, std::size_t>(queue,
                                                                   bytes));
    return alpaka::getPtrNative(*scratchBuf);
  }

  template <typename TIn, typename TOut>
  void convert(const TIn *input, TOut *output, std::size_t count) {
    if (count == 0)
      return;
    using Dimension = alpaka::DimInt<1u>;
    using Accelerator = alpaka::TagToAcc<TTag, Dimension, std::size_t>;
    const std::size_t threads = 256;
    const std::size_t blocks =
        std::min<std::size_t>((count + threads - 1) / threads, 65535);
    alpaka::exec<Accelerator>(queue,
                              alpaka::WorkDivMembers<Dimension, std::size_t>{
                                  blocks, threads, std::size_t{1}},
                              ConvertKernel{}, input, output, count);
  }

  template <typename TSum, typename TOut, typename TBias>
  void epilogue(const TSum *sums, TOut *C, unsigned int m, unsigned int n,
                int batchCount, long long strideC,
                EpilogueParams<TBias> const &params) {
    const std::size_t totalColumns = std::size_t{n} * batchCount;
    if (m == 0 || totalColumns == 0)
      return;
    const bool packed =
        m % 4 == 0 &&
        (!std::is_same_v<TOut, std::int8_t> ||
         (strideC % 4 == 0 && reinterpret_cast<std::uintptr_t>(C) % 4 == 0));
    using Dimension = alpaka::DimInt<2u>;
    using Accelerator = alpaka::TagToAcc<TTag, Dimension, std::size_t>;
    using Vector = alpaka::Vec<Dimension, std::size_t>;
    const std::size_t quadCount = (std::size_t{m} + 3) / 4;
    const std::size_t threadsX = std::min<std::size_t>(
        256, std::max<std::size_t>(32, (quadCount + 31) / 32 * 32));
    const std::size_t threadsY = 256 / threadsX;
    const Vector blocks{
        std::min<std::size_t>((totalColumns + threadsY - 1) / threadsY, 65535),
        (quadCount + threadsX - 1) / threadsX};
    withActivation(params.activation, [&](auto activation) {
      alpaka::exec<Accelerator>(queue,
                                alpaka::WorkDivMembers<Dimension, std::size_t>{
                                    blocks, Vector{threadsY, threadsX},
                                    Vector{std::size_t{1}, std::size_t{1}}},
                                EpilogueKernel{}, sums, C, std::size_t{m},
                                std::size_t{n}, totalColumns, strideC, params,
                                activation, packed);
    });
  }

  template <typename TFn>
  static void withActivation(Activation activation, TFn &&handler) {
    switch (activation) {
    case Activation::Relu:
      handler(ActivationTag<Activation::Relu>{});
      break;
    case Activation::Gelu:
      handler(ActivationTag<Activation::Gelu>{});
      break;
    default:
      handler(ActivationTag<Activation::None>{});
    }
  }

  using ScratchBuf = alpaka::Buf<alpaka::Dev<TQueue>, std::byte,
                                 alpaka::DimInt<1u>, std::size_t>;
  TQueue queue;
  std::optional<ScratchBuf> scratchBuf;
};
