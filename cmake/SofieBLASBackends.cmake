# Shared backend discovery for sofieBLAS tests and benchmarks.

include(CheckLanguage)
include(FetchContent)

set(CUDA_BASE "/usr/local/cuda" CACHE PATH "CUDA base path")
set(ROCM_BASE "/opt/rocm" CACHE PATH "ROCm base path")
set(ONEAPI_BASE "/opt/intel/oneapi" CACHE PATH "Intel oneAPI base path")
set(TBB_BASE "/usr" CACHE PATH "TBB base path")
set(BLIS_BASE "/usr" CACHE PATH "BLIS base path")

set(SOFIEBLAS_ALPAKA_GIT_TAG "2fa91a34ed11b2076e474c5507d920e85cf9b79d" CACHE STRING
    "alpaka commit to fetch when no installed alpaka is found")

find_package(alpaka CONFIG QUIET)
if(NOT alpaka_FOUND)
  message(STATUS "sofieBLAS: alpaka not found via find_package, fetching ${SOFIEBLAS_ALPAKA_GIT_TAG}")
  FetchContent_Declare(
    alpaka
    GIT_REPOSITORY https://github.com/alpaka-group/alpaka
    GIT_TAG        ${SOFIEBLAS_ALPAKA_GIT_TAG}
  )
  FetchContent_MakeAvailable(alpaka)
else()
  message(STATUS "sofieBLAS: using alpaka ${alpaka_VERSION} from ${alpaka_DIR}")
endif()

set(SOFIEBLAS_ENABLE_CUDA "AUTO" CACHE STRING "ON/OFF/AUTO: whether to enable the CUDA backend")
set(SOFIEBLAS_ENABLE_HIP "AUTO" CACHE STRING "ON/OFF/AUTO: whether to enable the HIP backend")
set_property(CACHE SOFIEBLAS_ENABLE_CUDA PROPERTY STRINGS ON OFF AUTO)
set_property(CACHE SOFIEBLAS_ENABLE_HIP PROPERTY STRINGS ON OFF AUTO)

function(sofieblas_detect_nvidia_gpu out)
  set(${out} FALSE PARENT_SCOPE)
  find_program(SOFIEBLAS_NVIDIA_SMI nvidia-smi HINTS ${CUDA_BASE}/bin)
  if(SOFIEBLAS_NVIDIA_SMI)
    execute_process(COMMAND ${SOFIEBLAS_NVIDIA_SMI} -L
      OUTPUT_VARIABLE _gpus RESULT_VARIABLE _result ERROR_QUIET)
    if(_result EQUAL 0 AND _gpus MATCHES "GPU [0-9]+:")
      set(${out} TRUE PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(sofieblas_detect_amd_gpus out_archs)
  set(${out_archs} "" PARENT_SCOPE)
  find_program(SOFIEBLAS_ROCM_AGENT_ENUMERATOR rocm_agent_enumerator
    HINTS ${ROCM_BASE}/bin)
  if(SOFIEBLAS_ROCM_AGENT_ENUMERATOR)
    execute_process(COMMAND ${SOFIEBLAS_ROCM_AGENT_ENUMERATOR}
      OUTPUT_VARIABLE _agents RESULT_VARIABLE _result ERROR_QUIET)
    if(_result EQUAL 0)
      string(REGEX MATCHALL "gfx[0-9a-f]+" _archs "${_agents}")
      list(REMOVE_ITEM _archs gfx000)
      list(REMOVE_DUPLICATES _archs)
      set(${out_archs} "${_archs}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

if(NOT SOFIEBLAS_ENABLE_CUDA STREQUAL "OFF")
  sofieblas_detect_nvidia_gpu(SOFIEBLAS_HAS_NVIDIA_GPU)
endif()
if(SOFIEBLAS_ENABLE_CUDA STREQUAL "OFF")
  message(STATUS "sofieBLAS: CUDA backend disabled (SOFIEBLAS_ENABLE_CUDA=OFF)")
elseif(SOFIEBLAS_ENABLE_CUDA STREQUAL "AUTO" AND NOT SOFIEBLAS_HAS_NVIDIA_GPU)
  message(STATUS "sofieBLAS: no NVIDIA GPU found, CUDA targets disabled (SOFIEBLAS_ENABLE_CUDA=ON builds them anyway)")
else()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    # Default to a portable build: 'all-major' emits SASS for every major arch the
    # installed nvcc supports plus PTX for the newest, so it never requests an arch
    # nvcc doesn't know and still JITs on newer GPUs.
    # Override with -DCMAKE_CUDA_ARCHITECTURES=native (or e.g. 80;90) for faster builds.
    if(SOFIEBLAS_HAS_NVIDIA_GPU AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      set(CMAKE_CUDA_ARCHITECTURES native)
    elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
      set(CMAKE_CUDA_ARCHITECTURES all-major)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 70 80)
    endif()
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(SOFIEBLAS_CUDA_ENABLED TRUE)
      message(STATUS "sofieBLAS: CUDA toolkit ${CUDAToolkit_VERSION} found, CUDA targets enabled for ${CMAKE_CUDA_ARCHITECTURES}")
    else()
      message(STATUS "sofieBLAS: nvcc found but CUDAToolkit package not found, CUDA targets disabled")
    endif()
  elseif(SOFIEBLAS_ENABLE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "sofieBLAS: SOFIEBLAS_ENABLE_CUDA=ON but no CUDA compiler was found")
  else()
    message(STATUS "sofieBLAS: no CUDA compiler found, CUDA targets disabled")
  endif()
endif()

if(NOT SOFIEBLAS_ENABLE_HIP STREQUAL "OFF")
  sofieblas_detect_amd_gpus(SOFIEBLAS_AMD_GPU_ARCHS)
endif()
if(SOFIEBLAS_ENABLE_HIP STREQUAL "OFF")
  message(STATUS "sofieBLAS: HIP backend disabled (SOFIEBLAS_ENABLE_HIP=OFF)")
elseif(SOFIEBLAS_ENABLE_HIP STREQUAL "AUTO" AND NOT SOFIEBLAS_AMD_GPU_ARCHS)
  message(STATUS "sofieBLAS: no AMD GPU found, HIP targets disabled (SOFIEBLAS_ENABLE_HIP=ON builds them anyway)")
else()
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    if(SOFIEBLAS_AMD_GPU_ARCHS)
      set(CMAKE_HIP_ARCHITECTURES ${SOFIEBLAS_AMD_GPU_ARCHS})
    else()
      set(CMAKE_HIP_ARCHITECTURES gfx1100)
    endif()
  endif()
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(SOFIEBLAS_HIP_ENABLED TRUE)

    # fp8 format of the HIP backend, a compile-time choice. 
    # The ROCm header is asked, per target, the format it supports.
    # FNUZ-only hardware (gfx942) takes FNUZ data from the caller,
    # OCP-only hardware (gfx950, gfx12) takes OCP. Targets without fp8 hardware
    # convert both in software and follow the others. FNUZ-only and OCP-only
    # targets cannot share a build.
    set(_sofieblas_targets ${CMAKE_HIP_ARCHITECTURES})
    if(_sofieblas_targets STREQUAL "native")
      set(_sofieblas_targets ${SOFIEBLAS_AMD_GPU_ARCHS})
    endif()
    set(_sofieblas_fnuz_targets "")
    set(_sofieblas_ocp_targets "")
    foreach(_arch IN LISTS _sofieblas_targets)
      foreach(_kind FNUZ OCP)
        execute_process(
          COMMAND ${CMAKE_HIP_COMPILER} -x hip --offload-arch=${_arch}
                  --offload-device-only -fsyntax-only -DSOFIEBLAS_PROBE_${_kind}_ONLY
                  ${CMAKE_CURRENT_LIST_DIR}/sofieblas_fp8_check.hip
          RESULT_VARIABLE _result OUTPUT_QUIET ERROR_QUIET)
        if(_result EQUAL 0)
          list(APPEND _sofieblas_${_kind}_ONLY_targets ${_arch})
        endif()
      endforeach()
    endforeach()
    set(_sofieblas_fnuz_targets ${_sofieblas_FNUZ_ONLY_targets})
    set(_sofieblas_ocp_targets ${_sofieblas_OCP_ONLY_targets})
    if(_sofieblas_fnuz_targets AND _sofieblas_ocp_targets)
      message(FATAL_ERROR "sofieBLAS: the HIP targets mix FNUZ-only fp8 (${_sofieblas_fnuz_targets}) "
        "with OCP-only fp8 (${_sofieblas_ocp_targets}). Build them separately.")
    elseif(_sofieblas_fnuz_targets)
      target_compile_definitions(sofieBLAS INTERFACE SOFIEBLAS_HIP_FP8_FNUZ=1)
      message(STATUS "sofieBLAS: HIP fp8 format is FNUZ (${_sofieblas_fnuz_targets})")
    else()
      message(STATUS "sofieBLAS: HIP fp8 format is OCP")
    endif()
    message(STATUS "sofieBLAS: HIP compiler found, HIP targets enabled for ${CMAKE_HIP_ARCHITECTURES}")
  elseif(SOFIEBLAS_ENABLE_HIP STREQUAL "ON")
    message(FATAL_ERROR "sofieBLAS: SOFIEBLAS_ENABLE_HIP=ON but no HIP compiler was found")
  else()
    message(STATUS "sofieBLAS: no HIP compiler found, HIP targets disabled")
  endif()
endif()

# --- CPU BLAS selection ---
set(AVAILABLE_BLAS_LIBS OpenBLAS MKL BLIS Accelerate CACHE STRING "Choose CPU BLAS library")
set_property(CACHE AVAILABLE_BLAS_LIBS PROPERTY STRINGS OpenBLAS MKL BLIS Accelerate)

if(NOT DEFINED CPU_BLAS_LIB)
  set(CPU_BLAS_LIB "OpenBLAS" CACHE STRING "CPU BLAS library to use")
endif()

function(sofieblas_find_openblas out_lib out_define out_include)
  find_library(OPENBLAS_LIB NAMES openblas
    PATHS /usr/lib/x86_64-linux-gnu/openblas-pthread
          /usr/lib/x86_64-linux-gnu/openblas-serial
          /usr/lib/x86_64-linux-gnu
          /usr/local/lib)
  find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h
    PATHS /usr/include
          /usr/include/x86_64-linux-gnu/openblas-pthread
          /usr/include/x86_64-linux-gnu/openblas-serial
          /usr/include/openblas
          /usr/local/include)
  if(OPENBLAS_LIB AND OPENBLAS_INCLUDE_DIR)
    set(${out_lib} ${OPENBLAS_LIB} PARENT_SCOPE)
    set(${out_define} SOFIEBLAS_USE_OPENBLAS PARENT_SCOPE)
    set(${out_include} ${OPENBLAS_INCLUDE_DIR} PARENT_SCOPE)
  else()
    set(${out_lib} "" PARENT_SCOPE)
    set(${out_define} "" PARENT_SCOPE)
    set(${out_include} "" PARENT_SCOPE)
  endif()
endfunction()

function(sofieblas_find_mkl out_lib out_define out_include)
  find_library(MKL_LIB NAMES mkl_rt PATHS ${ONEAPI_BASE}/mkl/latest/lib/intel64 ${ONEAPI_BASE}/mkl/latest/lib)
  find_path(MKL_INCLUDE_DIR NAMES mkl.h PATHS ${ONEAPI_BASE}/mkl/latest/include)
  if(MKL_LIB AND MKL_INCLUDE_DIR)
    set(${out_lib} ${MKL_LIB} PARENT_SCOPE)
    set(${out_define} SOFIEBLAS_USE_MKL PARENT_SCOPE)
    set(${out_include} ${MKL_INCLUDE_DIR} PARENT_SCOPE)
  else()
    set(${out_lib} "" PARENT_SCOPE)
    set(${out_define} "" PARENT_SCOPE)
    set(${out_include} "" PARENT_SCOPE)
  endif()
endfunction()

function(sofieblas_find_blis out_lib out_define out_include)
  find_library(BLIS_LIB NAMES blis PATHS ${BLIS_BASE}/lib ${BLIS_BASE}/lib64)
  find_path(BLIS_INCLUDE_DIR NAMES blis/cblas.h PATHS ${BLIS_BASE}/include)
  if(BLIS_LIB AND BLIS_INCLUDE_DIR)
    set(${out_lib} ${BLIS_LIB} PARENT_SCOPE)
    set(${out_define} SOFIEBLAS_USE_BLIS PARENT_SCOPE)
    set(${out_include} ${BLIS_INCLUDE_DIR} PARENT_SCOPE)
  else()
    set(${out_lib} "" PARENT_SCOPE)
    set(${out_define} "" PARENT_SCOPE)
    set(${out_include} "" PARENT_SCOPE)
  endif()
endfunction()

function(sofieblas_find_accelerate out_lib out_define out_include)
  set(${out_lib} "" PARENT_SCOPE)
  set(${out_define} "" PARENT_SCOPE)
  set(${out_include} "" PARENT_SCOPE)
  if(APPLE)
    find_library(ACCELERATE_LIB Accelerate)
    if(ACCELERATE_LIB)
      set(${out_lib} ${ACCELERATE_LIB} PARENT_SCOPE)
      set(${out_define} SOFIEBLAS_USE_ACCELERATE PARENT_SCOPE)
    endif()
  endif()
endfunction()

set(_search_order OpenBLAS MKL BLIS Accelerate)
if(NOT CPU_BLAS_LIB IN_LIST _search_order)
  message(FATAL_ERROR "Unknown CPU_BLAS_LIB option: ${CPU_BLAS_LIB}")
endif()
list(REMOVE_ITEM _search_order ${CPU_BLAS_LIB})
list(PREPEND _search_order ${CPU_BLAS_LIB})

foreach(_lib IN LISTS _search_order)
  if(NOT SOFIEBLAS_CPU_BLAS_LIBS)
    string(TOLOWER ${_lib} _lib_lower)
    cmake_language(CALL sofieblas_find_${_lib_lower}
      SOFIEBLAS_CPU_BLAS_LIBS SOFIEBLAS_CPU_BLAS_DEFINE SOFIEBLAS_CPU_BLAS_INCLUDE_DIR)
    if(SOFIEBLAS_CPU_BLAS_LIBS)
      set(SOFIEBLAS_CPU_BLAS_FOUND ${_lib})
    else()
      message(STATUS "sofieBLAS: ${_lib} not found, trying next CPU BLAS option...")
    endif()
  endif()
endforeach()

option(SOFIEBLAS_FETCH_OPENBLAS
  "Build OpenBLAS from source for the CPU targets when no CPU BLAS library is found" ON)
set(SOFIEBLAS_OPENBLAS_GIT_TAG "v0.3.28" CACHE STRING
    "OpenBLAS tag to build when no CPU BLAS library is found")
# Architecture-independent OpenBLAS configuration. Build-time CPU detection
# could fail on CPUs the pinned OpenBLAS does not know, leaving the GEMM
# unroll macros undefined. Where OpenBLAS supports
# DYNAMIC_ARCH we build all kernels and select one at run time (unknown CPUs
# fall back to the nearest kernel); elsewhere we use the portable GENERIC core.
# Set SOFIEBLAS_OPENBLAS_TARGET (e.g. HASWELL, ZEN) to force a specific core.
set(SOFIEBLAS_OPENBLAS_TARGET "" CACHE STRING
    "Force OpenBLAS TARGET core (empty = auto: DYNAMIC_ARCH or GENERIC)")
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _sb_proc)
set(_openblas_arch_args)
if(SOFIEBLAS_OPENBLAS_TARGET)
  list(APPEND _openblas_arch_args -DTARGET=${SOFIEBLAS_OPENBLAS_TARGET})
elseif(_sb_proc MATCHES "^(x86_64|amd64)$")
  # An explicit baseline TARGET is required: without it OpenBLAS probes the
  # build CPU, which fails for unrecognised CPUs.
  list(APPEND _openblas_arch_args -DDYNAMIC_ARCH=ON -DTARGET=GENERIC)
elseif(_sb_proc MATCHES "^(aarch64|arm64)$")
  list(APPEND _openblas_arch_args -DDYNAMIC_ARCH=ON -DTARGET=ARMV8)
elseif(_sb_proc MATCHES "^(ppc64le|powerpc64le)$")
  list(APPEND _openblas_arch_args -DDYNAMIC_ARCH=ON -DTARGET=POWER8)
else()
  list(APPEND _openblas_arch_args -DTARGET=GENERIC)
endif()

# Prebuilt OpenBLAS. OpenBLAS publishes no Linux binaries, so the wheels of the
# scipy-openblas32 PyPI package are used: they hold a DYNAMIC_ARCH build of
# OpenBLAS whose symbols carry a "scipy_" prefix.
# SOFIEBLAS_FETCH_PREBUILT_OPENBLAS=OFF skips it.
option(SOFIEBLAS_FETCH_PREBUILT_OPENBLAS
  "Download a prebuilt OpenBLAS when no CPU BLAS library is found" ON)
set(_sb_wheel_base "https://files.pythonhosted.org/packages")
set(_sb_wheel_url)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(_sb_proc MATCHES "^(x86_64|amd64)$")
    set(_sb_wheel_url "${_sb_wheel_base}/75/93/56ce22e41d59a45b885ac67f53425cd012329cb1e4de3285bebc4f8ad4bc/scipy_openblas32-0.3.34.481.3-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
    set(_sb_wheel_sha256 0595f94cd58c1f8a0d90c975621f4587337a003916b1979e834ebdf2cb5d9504)
  elseif(_sb_proc MATCHES "^(aarch64|arm64)$")
    set(_sb_wheel_url "${_sb_wheel_base}/04/3e/19a35f6b50552df2fa9464b526d0a4f98f4f9e7789b2adbb813d598640f7/scipy_openblas32-0.3.34.481.3-py3-none-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl")
    set(_sb_wheel_sha256 e085dddbc5b9aa130f83def515d31c440e69f387a90c7e63b154a5575b4466bc)
  elseif(_sb_proc MATCHES "^(ppc64le|powerpc64le)$")
    set(_sb_wheel_url "${_sb_wheel_base}/59/3d/5ebc0960f3ea86886e8b1ada3c12b589090de44db989fa303ea5181a61de/scipy_openblas32-0.3.34.481.3-py3-none-manylinux_2_27_ppc64le.manylinux_2_28_ppc64le.whl")
    set(_sb_wheel_sha256 f12ea9d388f887c10b921aaedaab861c957152b6b84d98e391a189ca7a256cbe)
  endif()
endif()

find_package(Threads REQUIRED)
if(NOT SOFIEBLAS_CPU_BLAS_LIBS AND SOFIEBLAS_FETCH_PREBUILT_OPENBLAS AND _sb_wheel_url)
  include(FetchContent)
  FetchContent_Declare(sofieblas_openblas_wheel
    URL ${_sb_wheel_url}
    URL_HASH SHA256=${_sb_wheel_sha256}
    DOWNLOAD_NO_EXTRACT TRUE)
  FetchContent_MakeAvailable(sofieblas_openblas_wheel)
  set(_sb_wheel_dir ${FETCHCONTENT_BASE_DIR}/sofieblas_openblas_wheel-extracted)
  if(NOT EXISTS ${_sb_wheel_dir}/scipy_openblas32/lib/libscipy_openblas.so)
    file(GLOB _sb_wheel_file ${sofieblas_openblas_wheel_SOURCE_DIR}/*.whl)
    file(ARCHIVE_EXTRACT INPUT ${_sb_wheel_file} DESTINATION ${_sb_wheel_dir})
  endif()
  set(_sb_prebuilt_lib ${_sb_wheel_dir}/scipy_openblas32/lib/libscipy_openblas.so)
  if(EXISTS ${_sb_prebuilt_lib})
    add_library(sofieblas_openblas INTERFACE)
    target_link_libraries(sofieblas_openblas INTERFACE
      ${_sb_prebuilt_lib} Threads::Threads m)
    # the prebuilt symbols are prefixed with scipy_; the code calls cblas_sgemm
    target_compile_definitions(sofieblas_openblas INTERFACE
      cblas_sgemm=scipy_cblas_sgemm)
    set(SOFIEBLAS_CPU_BLAS_LIBS sofieblas_openblas)
    set(SOFIEBLAS_CPU_BLAS_DEFINE SOFIEBLAS_USE_OPENBLAS)
    set(SOFIEBLAS_CPU_BLAS_INCLUDE_DIR ${_sb_wheel_dir}/scipy_openblas32/include)
    set(SOFIEBLAS_CPU_BLAS_FOUND "prebuilt OpenBLAS (scipy-openblas32 wheel)")
    message(STATUS "sofieBLAS: no CPU BLAS library found, using prebuilt OpenBLAS from the scipy-openblas32 wheel (SOFIEBLAS_FETCH_PREBUILT_OPENBLAS=OFF disables this)")
  else()
    message(STATUS "sofieBLAS: prebuilt OpenBLAS download unusable, falling back to a source build")
  endif()
endif()

if(NOT SOFIEBLAS_CPU_BLAS_LIBS AND SOFIEBLAS_FETCH_OPENBLAS)
  include(ExternalProject)
  set(_openblas_prefix ${CMAKE_BINARY_DIR}/_deps/openblas-install)
  set(_openblas_lib
    ${_openblas_prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  ExternalProject_Add(sofieblas_openblas_build
    GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS
    GIT_TAG ${SOFIEBLAS_OPENBLAS_GIT_TAG}
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE
    PREFIX ${CMAKE_BINARY_DIR}/_deps/openblas
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${_openblas_prefix}
      -DCMAKE_INSTALL_LIBDIR=lib
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DBUILD_SHARED_LIBS=OFF
      -DBUILD_STATIC_LIBS=ON
      -DBUILD_WITHOUT_LAPACK=ON
      -DNOFORTRAN=ON
      ${_openblas_arch_args}
      -DBUILD_TESTING=OFF
    BUILD_BYPRODUCTS ${_openblas_lib})
  add_library(sofieblas_openblas INTERFACE)
  target_link_libraries(sofieblas_openblas INTERFACE
    ${_openblas_lib} Threads::Threads m)
  add_dependencies(sofieblas_openblas sofieblas_openblas_build)
  set(SOFIEBLAS_CPU_BLAS_LIBS sofieblas_openblas)
  set(SOFIEBLAS_CPU_BLAS_DEFINE SOFIEBLAS_USE_OPENBLAS)
  set(SOFIEBLAS_CPU_BLAS_INCLUDE_DIR ${_openblas_prefix}/include/openblas)
  set(SOFIEBLAS_CPU_BLAS_FOUND "OpenBLAS ${SOFIEBLAS_OPENBLAS_GIT_TAG}, built from source")
  message(STATUS "sofieBLAS: no CPU BLAS library found (tried: ${_search_order}), building OpenBLAS ${SOFIEBLAS_OPENBLAS_GIT_TAG} from source (SOFIEBLAS_FETCH_OPENBLAS=OFF disables this)")
endif()

if(NOT SOFIEBLAS_CPU_BLAS_LIBS)
  message(WARNING "sofieBLAS: no suitable CPU BLAS library found (tried: ${_search_order}). CPU targets disabled.")
else()
  message(STATUS "sofieBLAS: using CPU BLAS library ${SOFIEBLAS_CPU_BLAS_FOUND} (${SOFIEBLAS_CPU_BLAS_LIBS})")
endif()
