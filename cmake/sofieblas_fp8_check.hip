// Compiled per HIP target by cmake/SofieBLASBackends.cmake, to read which fp8
// format the ROCm header says the device converts:
//   -DSOFIEBLAS_PROBE_FNUZ_ONLY  succeeds for FNUZ-only hardware (gfx942)
//   -DSOFIEBLAS_PROBE_OCP_ONLY   succeeds for OCP-only hardware (gfx950, gfx12)
// A target that passes neither has no fp8 hardware: the
// header offers both types there, with software conversions.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#if __HIP_DEVICE_COMPILE__
#if defined(SOFIEBLAS_PROBE_FNUZ_ONLY) && !(HIP_FP8_TYPE_FNUZ && !HIP_FP8_TYPE_OCP)
#error "not an FNUZ-only fp8 target"
#endif
#if defined(SOFIEBLAS_PROBE_OCP_ONLY) && !(HIP_FP8_TYPE_OCP && !HIP_FP8_TYPE_FNUZ)
#error "not an OCP-only fp8 target"
#endif
#endif

__global__ void sofieblas_fp8_probe() {}
