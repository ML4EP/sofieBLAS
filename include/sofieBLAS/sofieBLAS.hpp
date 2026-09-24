#pragma once

#include "sofieBLAS/core.hpp"

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
// If no CPU BLAS backend was selected on the compiler command line,
// default to OpenBLAS
#if !defined(SOFIEBLAS_USE_OPENBLAS) && !defined(SOFIEBLAS_USE_MKL) &&         \
    !defined(SOFIEBLAS_USE_BLIS) && !defined(SOFIEBLAS_USE_ACCELERATE)
#define SOFIEBLAS_USE_OPENBLAS
#endif

#if defined(SOFIEBLAS_USE_OPENBLAS)
#include "sofieBLAS/backends/cpu/sofieBLAS_openblas.hpp"
#elif defined(SOFIEBLAS_USE_MKL)
#include "sofieBLAS/backends/cpu/sofieBLAS_mkl.hpp"
#elif defined(SOFIEBLAS_USE_BLIS)
#include "sofieBLAS/backends/cpu/sofieBLAS_blis.hpp"
#elif defined(SOFIEBLAS_USE_ACCELERATE)
#include "sofieBLAS/backends/cpu/sofieBLAS_accelerate.hpp"
#endif
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include "sofieBLAS/backends/cuda/sofieBLAS_cublas.hpp"
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include "sofieBLAS/backends/hip/sofieBLAS_hipblaslt.hpp"
#endif
