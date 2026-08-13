cmake_minimum_required(VERSION 3.21)
project(AlpakaTest LANGUAGES CXX)

# --- User-configurable options (override via -D) ---
set(CUDA_BASE "/usr/local/cuda" CACHE PATH "CUDA base path")
set(ROCM_BASE "/opt/rocm" CACHE PATH "ROCm base path")
set(ONEAPI_BASE "/opt/intel/oneapi" CACHE PATH "Intel oneAPI base path")
set(TBB_BASE "/usr" CACHE PATH "TBB base path")
set(ALPAKA_BASE "$ENV{HOME}/src/alpaka-group/alpaka" CACHE PATH "Alpaka base path")

# --- CPU BLAS selection ---
set(AVAILABLE_BLAS_LIBS OpenBLAS MKL BLIS Accelerate CACHE STRING "Choose CPU BLAS library")
set_property(CACHE AVAILABLE_BLAS_LIBS PROPERTY STRINGS OpenBLAS MKL BLIS Accelerate)

if(NOT DEFINED CPU_BLAS_LIB)
  set(CPU_BLAS_LIB "OpenBLAS" CACHE STRING "CPU BLAS library to use")
endif()

set(BLIS_BASE "/usr" CACHE PATH "BLIS base path")

# --- Compiler flags ---
set(CXXFLAGS -O2 -g -DALPAKA_HAS_STD_ATOMIC_REF)
set(CXX_HOST_FLAGS -fPIC -pthread)
set(CXX_CUDA_FLAGS -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr)
set(XCOMPILER_FLAGS -Xcompiler=-fPIC,-pthread)

include(CheckLanguage)

# --- CUDA architecture: must be set before enable_language(CUDA) so all
#     targets inherit a valid default (CMake 3.18+ requires CUDA_ARCHITECTURES
#     to be non-empty on every target once CUDA is enabled globally). Defaults
#     to the architecture of the GPU in the build machine; override with
#     -DCMAKE_CUDA_ARCHITECTURES=<arch> to build for a different one.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(STATUS "No CUDA compiler found, test_cuda target will not be built")
endif()

# --- HIP architecture: CMAKE_HIP_ARCHITECTURES is left unset on purpose, so
#     CMake initializes it from rocm_agent_enumerator, i.e. the GPUs actually
#     installed. Override with -DCMAKE_HIP_ARCHITECTURES=<arch> if needed.
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
else()
  message(STATUS "No HIP compiler found, test_hip target will not be built")
endif()

# --- Include directories ---
include_directories(${ALPAKA_BASE}/include "../include")

# --- Functions to find BLAS libraries ---
function(find_openblas blas_lib blas_define blas_include)
  find_path(OPENBLAS_PATH NAMES libopenblas.a libopenblas.so libopenblas.so.0
    PATHS /usr/lib64 /usr/lib/x86_64-linux-gnu/openblas-serial /usr/lib/x86_64-linux-gnu)
  if(OPENBLAS_PATH)
    find_library(OPENBLAS_LIB NAMES openblas libopenblas.so.0 PATHS ${OPENBLAS_PATH})
    find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h
      PATHS /usr/include /usr/include/openblas /usr/include/cblas)
    set(${blas_lib} ${OPENBLAS_LIB} PARENT_SCOPE)
    set(${blas_define} SOFIEBLAS_USE_OPENBLAS PARENT_SCOPE)
    set(${blas_include} ${OPENBLAS_INCLUDE_DIR} PARENT_SCOPE)
  else()
    set(${blas_lib} "" PARENT_SCOPE)
    set(${blas_define} "" PARENT_SCOPE)
    set(${blas_include} "" PARENT_SCOPE)
  endif()
endfunction()

function(find_mkl blas_lib blas_define blas_include)
  find_path(MKL_PATH NAMES libmkl_rt.a PATHS ${ONEAPI_BASE}/mkl/latest/lib/intel64)
  if(MKL_PATH)
    find_library(MKL_LIB mkl_rt PATHS ${MKL_PATH})
    set(${blas_lib} ${MKL_LIB} PARENT_SCOPE)
    set(${blas_define} SOFIEBLAS_USE_MKL PARENT_SCOPE)
    set(${blas_include} ${ONEAPI_BASE}/mkl/latest/include PARENT_SCOPE)
  else()
    set(${blas_lib} "" PARENT_SCOPE)
    set(${blas_define} "" PARENT_SCOPE)
    set(${blas_include} "" PARENT_SCOPE)
  endif()
endfunction()

function(find_blis blas_lib blas_define blas_include)
  find_path(BLIS_PATH NAMES libblis.a libblis.so
    PATHS ${BLIS_BASE}/lib ${BLIS_BASE}/lib64)
  if(BLIS_PATH)
    find_library(BLIS_LIB blis PATHS ${BLIS_PATH})
    find_path(BLIS_INCLUDE_DIR NAMES blis/cblas.h PATHS ${BLIS_BASE}/include)
    set(${blas_lib} ${BLIS_LIB} PARENT_SCOPE)
    set(${blas_define} SOFIEBLAS_USE_BLIS PARENT_SCOPE)
    set(${blas_include} ${BLIS_INCLUDE_DIR} PARENT_SCOPE)
  else()
    set(${blas_lib} "" PARENT_SCOPE)
    set(${blas_define} "" PARENT_SCOPE)
    set(${blas_include} "" PARENT_SCOPE)
  endif()
endfunction()

function(find_accelerate blas_lib blas_define blas_include)
  set(${blas_lib} "" PARENT_SCOPE)
  set(${blas_define} "" PARENT_SCOPE)
  set(${blas_include} "" PARENT_SCOPE)
  if(APPLE)
    find_library(ACCELERATE_LIB Accelerate)
    if(ACCELERATE_LIB)
      set(${blas_lib} ${ACCELERATE_LIB} PARENT_SCOPE)
      set(${blas_define} SOFIEBLAS_USE_ACCELERATE PARENT_SCOPE)
    endif()
  endif()
endfunction()

# --- BLAS detection and selection ---
set(SOFIEBLAS_BLAS_SEARCH_ORDER OpenBLAS MKL BLIS Accelerate)
if(NOT CPU_BLAS_LIB IN_LIST SOFIEBLAS_BLAS_SEARCH_ORDER)
  message(FATAL_ERROR "Unknown CPU_BLAS_LIB option: ${CPU_BLAS_LIB}")
endif()
list(REMOVE_ITEM SOFIEBLAS_BLAS_SEARCH_ORDER ${CPU_BLAS_LIB})
list(PREPEND SOFIEBLAS_BLAS_SEARCH_ORDER ${CPU_BLAS_LIB})

foreach(lib IN LISTS SOFIEBLAS_BLAS_SEARCH_ORDER)
  if(NOT BLAS_LIBS)
    string(TOLOWER ${lib} lib_lower)
    cmake_language(CALL find_${lib_lower} BLAS_LIBS SOFIEBLAS_DEFINES BLAS_INCLUDE_DIR)
    if(BLAS_LIBS)
      set(FOUND_BLAS_LIB ${lib})
    else()
      message(WARNING "${lib} not found, trying next option...")
    endif()
  endif()
endforeach()

if(NOT BLAS_LIBS)
  message(FATAL_ERROR "No suitable CPU BLAS library found! Tried: ${SOFIEBLAS_BLAS_SEARCH_ORDER}")
endif()

message(STATUS "Using CPU BLAS library: ${FOUND_BLAS_LIB}")
message(STATUS "BLAS libraries: ${BLAS_LIBS}")
message(STATUS "BLAS include dir: ${BLAS_INCLUDE_DIR}")
message(STATUS "Compile defines: ${SOFIEBLAS_DEFINES}")

# --- test_cpu target ---
add_executable(test_cpu test.cc)
target_compile_features(test_cpu PUBLIC cxx_std_20)
target_compile_options(test_cpu PRIVATE ${CXXFLAGS} ${CXX_HOST_FLAGS})
target_compile_definitions(test_cpu PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED ${SOFIEBLAS_DEFINES})
target_include_directories(test_cpu PRIVATE "../sofieBLAS/include" ${ALPAKA_BASE}/include ${BLAS_INCLUDE_DIR})
target_link_libraries(test_cpu PRIVATE ${BLAS_LIBS})


# --- test_cuda target ---
if(CMAKE_CUDA_COMPILER)
  add_executable(test_cuda)
  target_sources(test_cuda PRIVATE test.cc)
  set_source_files_properties(test.cc PROPERTIES LANGUAGE CUDA)
  set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_features(test_cuda PUBLIC cxx_std_20)

  target_compile_options(test_cuda PRIVATE
    ${CXXFLAGS}
    ${CXX_CUDA_FLAGS}
    ${XCOMPILER_FLAGS}
  )

  target_compile_definitions(test_cuda PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
  target_include_directories(test_cuda PRIVATE ${ALPAKA_BASE}/include "../sofieBLAS/include" ${CUDA_BASE}/include)
  target_link_directories(test_cuda PRIVATE ${CUDA_BASE}/lib64)
  target_link_libraries(test_cuda PRIVATE cublasLt cublas cudart)
endif()

# --- test_hip target ---
if(CMAKE_HIP_COMPILER)
  # Use a private copy of the source: set_source_files_properties(LANGUAGE ...)
  # is directory-scoped, so reusing test.cc directly here would clobber the
  # LANGUAGE CUDA property already set on it for test_cuda above.
  configure_file(test.cc test_hip_gen.cc COPYONLY)
  add_executable(test_hip)
  target_sources(test_hip PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/test_hip_gen.cc)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/test_hip_gen.cc PROPERTIES LANGUAGE HIP)
  target_compile_features(test_hip PUBLIC cxx_std_20)

  target_compile_options(test_hip PRIVATE
    ${CXXFLAGS}
    ${CXX_HOST_FLAGS}
  )

  target_compile_definitions(test_hip PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED)
  target_include_directories(test_hip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ALPAKA_BASE}/include "../sofieBLAS/include" ${ROCM_BASE}/include)
  target_link_directories(test_hip PRIVATE ${ROCM_BASE}/lib)
  target_link_libraries(test_hip PRIVATE hipblaslt hipblas amdhip64)
endif()

# --- clean target equivalent ---
add_custom_target(clean-all
  COMMAND ${CMAKE_COMMAND} -E rm -f test_cpu test_cuda test_hip *.d *.o *.so
  COMMENT "Cleaning all generated files"
)
