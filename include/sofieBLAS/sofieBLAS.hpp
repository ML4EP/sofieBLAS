#pragma once

#include "sofieBLAS/core.hpp"

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
#include "sofieBLAS/backends/cpu/sofieBLAS_cpu.hpp"
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include "sofieBLAS/backends/cuda/sofieBLAS_cublas.hpp"
#endif
