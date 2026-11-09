#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>

// fp8 elements are handled by the type's own conversions, in the format of the
// backend: the caller is responsible for providing data in that format
template <typename TOut, typename TIn>
ALPAKA_FN_HOST_ACC TOut convertElement(TIn const &element) {
  return static_cast<TOut>(element);
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
    float combinedScale = alpha;
    if (scaleA)
      combinedScale *= *scaleA;
    if (scaleB)
      combinedScale *= *scaleB;
    return combinedScale;
  }
};

template <typename TOut, Activation Act, typename TAcc, typename TSum,
          typename TBias>
ALPAKA_FN_ACC TOut applyEpilogue(TAcc const &accelerator, TSum sumVal,
                                 std::size_t i,
                                 EpilogueParams<TBias> const &params,
                                 float scale) {
  if constexpr (std::is_same_v<TOut, std::int32_t>) {
    return static_cast<std::int32_t>(sumVal);
  } else {
    float value = (params.alphaVector ? params.alphaVector[i] : scale) *
                  static_cast<float>(sumVal);
    if (params.bias)
      value += static_cast<float>(params.bias[i]);
    if constexpr (Act == Activation::Relu)
      value = value > 0.f ? value : 0.f;
    else if constexpr (Act == Activation::Gelu)
      value *= 0.5f * (1.f + alpaka::math::tanh(
                                 accelerator, 0.7978845608028654f *
                                                  (value + 0.044715f * value *
                                                               value * value)));
    if constexpr (std::is_same_v<TOut, std::int8_t>) {
      value = ::rintf(value);
      return static_cast<std::int8_t>(value > 127.f    ? 127.f
                                      : value < -128.f ? -128.f
                                                       : value);
    } else {
      return convertElement<TOut>(params.scaleD ? value * *params.scaleD
                                                : value);
    }
  }
}

struct ConvertKernel {
  template <typename TAcc, typename TIn, typename TOut>
  ALPAKA_FN_ACC void operator()(TAcc const &accelerator, const TIn *input,
                                TOut *output, std::size_t count) const {
    for (auto elementIdx : alpaka::uniformElements(accelerator, count))
      output[elementIdx] = convertElement<TOut>(input[elementIdx]);
  }
};

// the anonymous union keeps T's (host-only, for fp8) default constructor out of
// the device code
template <typename T, std::size_t N> struct alignas(sizeof(T) * N) Packed {
  union {
    T values[N];
  };
  ALPAKA_FN_HOST_ACC Packed() {}
};

struct EpilogueKernel {
  template <typename TAcc, typename TSum, typename TOut, typename TBias,
            Activation Act>
  ALPAKA_FN_ACC void operator()(TAcc const &accelerator, const TSum *sumMatrix,
                                TOut *output, std::size_t m, std::size_t n,
                                std::size_t totalColumns, long long strideOut,
                                EpilogueParams<TBias> params,
                                ActivationTag<Act>, bool packed) const {
    const float scale = params.scale();
    const std::size_t quads = (m + 3) / 4;
    for (auto col : alpaka::uniformElementsAlongY(accelerator, totalColumns)) {
      const std::size_t batchIdx = col / n;
      const TSum *sumColumn = sumMatrix + col * m;
      TOut *outputColumn = output +
                           static_cast<long long>(batchIdx) * strideOut +
                           (col - batchIdx * n) * m;
      for (auto quadIdx : alpaka::uniformElementsAlongX(accelerator, quads)) {
        const std::size_t firstRow = quadIdx * 4;
        Packed<TSum, 4> inputQuad;
        if (packed)
          inputQuad =
              *reinterpret_cast<const Packed<TSum, 4> *>(sumColumn + firstRow);
        else
          for (std::size_t quadPosition = 0; quadPosition < 4; ++quadPosition)
            inputQuad.values[quadPosition] =
                firstRow + quadPosition < m ? sumColumn[firstRow + quadPosition]
                                            : TSum{};
        Packed<TOut, 4> resultQuad;
        for (std::size_t quadPosition = 0; quadPosition < 4; ++quadPosition)
          resultQuad.values[quadPosition] = applyEpilogue<TOut, Act>(
              accelerator, inputQuad.values[quadPosition],
              firstRow + quadPosition < m ? firstRow + quadPosition : m - 1,
              params, scale);
        if (packed && std::is_same_v<TOut, std::int8_t>)
          *reinterpret_cast<Packed<TOut, 4> *>(outputColumn + firstRow) =
              resultQuad;
        else
          for (std::size_t quadPosition = 0;
               quadPosition < 4 && firstRow + quadPosition < m; ++quadPosition)
            outputColumn[firstRow + quadPosition] =
                resultQuad.values[quadPosition];
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
  ALPAKA_FN_ACC void
  operator()(TAcc const &accelerator, TSum, const TA *A, const TB *B, TOut *C,
             std::size_t m, std::size_t n, std::size_t k, bool transA,
             bool transB, long long strideA, long long strideB,
             long long strideC, EpilogueParams<TBias> params,
             ActivationTag<Act>) const {
    auto &tileA =
        alpaka::declareSharedVar<TSum[depth][tile], __COUNTER__>(accelerator);
    auto &tileB =
        alpaka::declareSharedVar<TSum[depth][tile], __COUNTER__>(accelerator);
    const auto block =
        alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(accelerator);
    const auto thread =
        alpaka::getIdx<alpaka::Block, alpaka::Threads>(accelerator);
    const std::size_t batch = block[0];
    const std::size_t firstColumn = block[1] * tile, firstRow = block[2] * tile;
    const std::size_t threadY = thread[1], threadX = thread[2];
    const std::size_t threadLinearIdx = threadY * threads + threadX;
    const TA *batchA = A + static_cast<long long>(batch) * strideA;
    const TB *batchB = B + static_cast<long long>(batch) * strideB;
    const std::size_t lda = transA ? k : m, ldb = transB ? n : k;

    TSum accumulators[perThread][perThread] = {};
    for (std::size_t firstDepth = 0; firstDepth < k; firstDepth += depth) {
      for (std::size_t loadIdx = threadLinearIdx; loadIdx < depth * tile;
           loadIdx += threads * threads) {
        const std::size_t rowInTile = loadIdx % tile,
                          depthIdxA = loadIdx / tile;
        const std::size_t i = firstRow + rowInTile,
                          globalDepthA = firstDepth + depthIdxA;
        tileA[depthIdxA][rowInTile] =
            i < m && globalDepthA < k
                ? convertElement<TSum>(batchA[transA ? globalDepthA + i * lda
                                                     : i + globalDepthA * lda])
                : TSum{};
        const std::size_t depthIdxB = loadIdx % depth,
                          columnInTile = loadIdx / depth;
        const std::size_t j = firstColumn + columnInTile,
                          globalDepthB = firstDepth + depthIdxB;
        tileB[depthIdxB][columnInTile] =
            j < n && globalDepthB < k
                ? convertElement<TSum>(batchB[transB ? j + globalDepthB * ldb
                                                     : globalDepthB + j * ldb])
                : TSum{};
      }
      alpaka::syncBlockThreads(accelerator);
      for (std::size_t p = 0; p < depth; ++p) {
        TSum fragmentA[perThread], fragmentB[perThread];
        for (std::size_t registerRow = 0; registerRow < perThread;
             ++registerRow) {
          fragmentA[registerRow] = tileA[p][threadX + threads * registerRow];
          fragmentB[registerRow] = tileB[p][threadY + threads * registerRow];
        }
        for (std::size_t registerRow = 0; registerRow < perThread;
             ++registerRow)
          for (std::size_t registerColumn = 0; registerColumn < perThread;
               ++registerColumn)
            accumulators[registerRow][registerColumn] +=
                fragmentA[registerRow] * fragmentB[registerColumn];
      }
      alpaka::syncBlockThreads(accelerator);
    }

    const float scale = params.scale();
    TOut *output = C + static_cast<long long>(batch) * strideC;
    for (std::size_t registerRow = 0; registerRow < perThread; ++registerRow)
      for (std::size_t registerColumn = 0; registerColumn < perThread;
           ++registerColumn) {
        const std::size_t i = firstRow + threadX + threads * registerRow;
        const std::size_t j = firstColumn + threadY + threads * registerColumn;
        if (i < m && j < n)
          output[i + j * m] = applyEpilogue<TOut, Act>(
              accelerator, accumulators[registerRow][registerColumn], i, params,
              scale);
      }
  }
};
