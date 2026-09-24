#pragma once

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) &&                             \
    defined(SOFIEBLAS_USE_ACCELERATE)

// Apple's Accelerate framework (macOS only) exposes the same CBLAS API.
#include <Accelerate/Accelerate.h>

#include "sofieBLAS/backends/cpu/detail/sofieBLAS_cblas_common.hpp"

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED && SOFIEBLAS_USE_ACCELERATE
