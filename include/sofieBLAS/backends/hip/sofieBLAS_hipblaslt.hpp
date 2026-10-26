#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include "sofieBLAS/backends/gpu/detail/sofieBLAS_blaslt_common.tpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#define CHECK_HIP(err)                                                         \
  if ((err) != hipSuccess) {                                                   \
    std::cerr << "HIP error: " << hipGetErrorString(err) << " at line "        \
              << __LINE__ << "\n";                                             \
    exit(EXIT_FAILURE);                                                        \
  }

#define CHECK_HIPBLAS(status)                                                  \
  do {                                                                         \
    hipblasStatus_t _s = (status);                                             \
    if (_s != HIPBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "hipBLAS error " << _s << " at line " << __LINE__ << "\n";  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// The hipBLASLt forwarding of the shared BlasLt implementation in
// backends/gpu/detail
struct HipblasLtApi {
  using Queue = alpaka::QueueHipRtNonBlocking;
  using Handle = hipblasLtHandle_t;
  using BlasHandle = hipblasHandle_t;
  using Preference = hipblasLtMatmulPreference_t;
  using Stream = hipStream_t;
  using Layout = hipblasLtMatrixLayout_t;
  using MatmulDesc = hipblasLtMatmulDesc_t;
  using HeuristicResult = hipblasLtMatmulHeuristicResult_t;
  using Operation = hipblasOperation_t;
  using Epilogue = hipblasLtEpilogue_t;
  using Bf16 = __hip_bfloat16;
  using Fp8E4M3 = __hip_fp8_e4m3;
  using Fp8E5M2 = __hip_fp8_e5m2;

  static constexpr auto OpN = HIPBLAS_OP_N;
  static constexpr auto OpT = HIPBLAS_OP_T;
  static constexpr auto OpC = HIPBLAS_OP_C;
  static constexpr auto EpilogueDefault = HIPBLASLT_EPILOGUE_DEFAULT;
  static constexpr auto EpilogueBias = HIPBLASLT_EPILOGUE_BIAS;
  static constexpr auto EpilogueReluBias = HIPBLASLT_EPILOGUE_RELU_BIAS;
  static constexpr auto EpilogueGeluBias = HIPBLASLT_EPILOGUE_GELU_BIAS;
  static constexpr auto EpilogueRelu = HIPBLASLT_EPILOGUE_RELU;
  static constexpr auto ComputeF32 = HIPBLAS_COMPUTE_32F;
  static constexpr auto ComputeI32 = HIPBLAS_COMPUTE_32I;
  static constexpr auto RealF32 = HIP_R_32F;
  static constexpr auto RealF16 = HIP_R_16F;
  static constexpr auto RealBF16 = HIP_R_16BF;
  static constexpr auto RealF8E4M3 = HIP_R_8F_E4M3;
  static constexpr auto RealF8E5M2 = HIP_R_8F_E5M2;
  static constexpr auto RealI8 = HIP_R_8I;
  static constexpr auto RealI32 = HIP_R_32I;
  static constexpr auto DescTransA = HIPBLASLT_MATMUL_DESC_TRANSA;
  static constexpr auto DescTransB = HIPBLASLT_MATMUL_DESC_TRANSB;
  static constexpr auto DescEpilogue = HIPBLASLT_MATMUL_DESC_EPILOGUE;
  static constexpr auto DescBiasPointer = HIPBLASLT_MATMUL_DESC_BIAS_POINTER;
  static constexpr auto DescBiasDataType = HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE;
  static constexpr auto DescAScalePointer =
      HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
  static constexpr auto DescBScalePointer =
      HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;
  static constexpr auto DescDScalePointer =
      HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER;
  // The per-row alpha vector pointer mode and the bias batch stride of
  // cuBLASLt (PointerModeAlphaDeviceVectorBetaHost, DescBiasBatchStride) are
  // not mapped for hipBLASLt. The shared code checks whether the table has
  // them: int8MatmulVectorAlpha throws "unsupported" here, and a batched
  // bias with its own stride is issued one batch at a time.
  static constexpr auto LayoutBatchCount = HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT;
  static constexpr auto LayoutStridedBatchOffset =
      HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET;
  static constexpr auto PrefMaxWorkspace =
      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;
  static constexpr auto StatusNotSupported = HIPBLAS_STATUS_NOT_SUPPORTED;
  static constexpr const char *name = "hipBLASLt";

  static constexpr auto ltCreate = hipblasLtCreate;
  static constexpr auto ltDestroy = hipblasLtDestroy;
  static constexpr auto blasCreate = hipblasCreate;
  static constexpr auto blasDestroy = hipblasDestroy;
  static constexpr auto blasSetStream = hipblasSetStream;
  static constexpr auto prefCreate = hipblasLtMatmulPreferenceCreate;
  static constexpr auto prefDestroy = hipblasLtMatmulPreferenceDestroy;
  static constexpr auto prefSetAttribute =
      hipblasLtMatmulPreferenceSetAttribute;
  static constexpr auto layoutCreate = hipblasLtMatrixLayoutCreate;
  static constexpr auto layoutDestroy = hipblasLtMatrixLayoutDestroy;
  static constexpr auto layoutSetAttribute = hipblasLtMatrixLayoutSetAttribute;
  static constexpr auto descCreate = hipblasLtMatmulDescCreate;
  static constexpr auto descDestroy = hipblasLtMatmulDescDestroy;
  static constexpr auto descSetAttribute = hipblasLtMatmulDescSetAttribute;
  static constexpr auto getHeuristic = hipblasLtMatmulAlgoGetHeuristic;
  static constexpr auto matmul = hipblasLtMatmul;
  static constexpr auto sgemmStridedBatched = hipblasSgemmStridedBatched;

  // hipMalloc has a templated C++ overload, so a pointer to it is ambiguous
  static hipError_t rtMalloc(void **ptr, std::size_t size) {
    return hipMalloc(ptr, size);
  }
  static hipError_t rtFree(void *ptr) { return hipFree(ptr); }
};

#define SOFIEBLAS_CHECK_LT(status) CHECK_HIPBLAS(status)
#define SOFIEBLAS_CHECK_RT(err) CHECK_HIP(err)

using BlasHip = BlasLt<HipblasLtApi>;

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuHipRt> {
public:
  using Impl = BlasHip;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
