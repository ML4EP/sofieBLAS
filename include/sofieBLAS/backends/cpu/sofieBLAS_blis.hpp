#pragma once

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) && defined(SOFIEBLAS_USE_BLIS)

// BLIS installs its CBLAS-compatible header under blis/cblas.h
// when configured with --enable-cblas.
#include <blis/cblas.h>

#include "sofieBLAS/backends/cpu/detail/sofieBLAS_cblas_common.hpp"

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED && SOFIEBLAS_USE_BLIS
