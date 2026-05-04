cmake_minimum_required(VERSION 3.21)
project(AlpakaTest LANGUAGES CXX)

# --- User-configurable options (override via -D) ---
set(CUDA_BASE "/usr/local/cuda" CACHE PATH "CUDA base path")
set(ROCM_BASE "/opt/rocm" CACHE PATH "ROCm base path")
set(ONEAPI_BASE "/opt/intel/oneapi" CACHE PATH "Intel oneAPI base path")
set(TBB_BASE "/usr" CACHE PATH "TBB base path")
set(ALPAKA_BASE "$ENV{HOME}/src/alpaka-group/alpaka" CACHE PATH "Alpaka base path")

# --- CPU BLAS selection ---
set(AVAILABLE_BLAS_LIBS OpenBLAS MKL CACHE STRING "Choose CPU BLAS library")
set_property(CACHE AVAILABLE_BLAS_LIBS PROPERTY STRINGS OpenBLAS MKL)

if(NOT DEFINED CPU_BLAS_LIB)
  set(CPU_BLAS_LIB "OpenBLAS" CACHE STRING "CPU BLAS library to use")
endif()

# --- Compiler flags ---
set(CXXFLAGS -O2 -g -DALPAKA_HAS_STD_ATOMIC_REF)
set(CXX_HOST_FLAGS -fPIC -pthread)
set(CXX_CUDA_FLAGS -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr)
set(XCOMPILER_FLAGS -Xcompiler=-fPIC,-pthread)

# --- CUDA architecture: must be set before enable_language(CUDA) so all
#     targets inherit a valid default (CMake 3.18+ requires CUDA_ARCHITECTURES
#     to be non-empty on every target once CUDA is enabled globally).
set(CMAKE_CUDA_ARCHITECTURES 86)
enable_language(CUDA)

# --- Include directories ---
include_directories(${ALPAKA_BASE}/include "../include")

# --- Functions to find BLAS libraries ---
function(find_openblas blas_lib blas_define)
  find_path(OPENBLAS_PATH NAMES libopenblas.a
    PATHS /usr/lib/x86_64-linux-gnu/openblas-serial /usr/lib/x86_64-linux-gnu)
  if(OPENBLAS_PATH)
    find_library(OPENBLAS_LIB openblas PATHS ${OPENBLAS_PATH})
    set(${blas_lib} ${OPENBLAS_LIB} PARENT_SCOPE)
    set(${blas_define} SOFIEBLAS_USE_OPENBLAS PARENT_SCOPE)
  else()
    set(${blas_lib} "" PARENT_SCOPE)
    set(${blas_define} "" PARENT_SCOPE)
  endif()
endfunction()

function(find_mkl blas_lib blas_define)
  find_path(MKL_PATH NAMES libmkl_rt.a PATHS ${ONEAPI_BASE}/mkl/latest/lib/intel64)
  if(MKL_PATH)
    find_library(MKL_LIB mkl_rt PATHS ${MKL_PATH})
    set(${blas_lib} ${MKL_LIB} PARENT_SCOPE)
    set(${blas_define} SOFIEBLAS_USE_MKL PARENT_SCOPE)
  else()
    set(${blas_lib} "" PARENT_SCOPE)
    set(${blas_define} "" PARENT_SCOPE)
  endif()
endfunction()

# --- BLAS detection and selection ---
if(CPU_BLAS_LIB STREQUAL "OpenBLAS")
  find_openblas(BLAS_LIBS SOFIEBLAS_DEFINES)
  if(NOT BLAS_LIBS)
    message(WARNING "OpenBLAS not found, trying MKL...")
    find_mkl(BLAS_LIBS SOFIEBLAS_DEFINES)
  endif()
elseif(CPU_BLAS_LIB STREQUAL "MKL")
  find_mkl(BLAS_LIBS SOFIEBLAS_DEFINES)
  if(NOT BLAS_LIBS)
    message(WARNING "MKL not found, trying OpenBLAS...")
    find_openblas(BLAS_LIBS SOFIEBLAS_DEFINES)
  endif()
else()
  message(FATAL_ERROR "Unknown CPU_BLAS_LIB option: ${CPU_BLAS_LIB}")
endif()

if(NOT BLAS_LIBS)
  message(FATAL_ERROR "No suitable CPU BLAS library found!")
endif()

message(STATUS "Using CPU BLAS library: ${CPU_BLAS_LIB}")
message(STATUS "BLAS libraries: ${BLAS_LIBS}")
message(STATUS "Compile defines: ${SOFIEBLAS_DEFINES}")

# --- test_cpu target ---
add_executable(test_cpu test.cc)
target_compile_features(test_cpu PUBLIC cxx_std_20)
target_compile_options(test_cpu PRIVATE ${CXXFLAGS} ${CXX_HOST_FLAGS})
target_compile_definitions(test_cpu PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED ${SOFIEBLAS_DEFINES})
target_include_directories(test_cpu PRIVATE "../sofieBLAS/include" ${ALPAKA_BASE}/include)
target_link_libraries(test_cpu PRIVATE ${BLAS_LIBS})


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

# --- clean target equivalent ---
add_custom_target(clean-all
  COMMAND ${CMAKE_COMMAND} -E rm -f test_cpu test_cuda *.d *.o *.so
  COMMENT "Cleaning all generated files"
)
