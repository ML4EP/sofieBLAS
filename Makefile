.PHONY: all clean

all: test_cpu test_cuda

clean:
	rm -f test_cpu test_cuda *.d *.o *.so

# NVIDIA CUDA
CUDA_BASE   ?= /usr/local/cuda

# AMD HIP/ROCm
ROCM_BASE   ?= /opt/rocm

# Intel oneAPI
ONEAPI_BASE ?= /opt/intel/oneapi

# Intel TBB library
TBB_BASE    ?= /usr

# Alpaka library
ALPAKA_BASE ?= $(HOME)/src/alpaka-group/alpaka

CXX  := g++
NVCC := $(CUDA_BASE)/bin/nvcc
HIPCC := $(ROCM_BASE)/bin/hipcc
ICPX := $(ONEAPI_BASE)/compiler/latest/bin/icpx

CUDA_ARCH      := sm_86
ROCM_ARCH      := gfx1100
SYCL_GPU_ARCH  := intel_gpu_tgllp

CXXFLAGS       := -std=c++20 -O2 -g -I$(ALPAKA_BASE)/include -DALPAKA_HAS_STD_ATOMIC_REF
CXX_HOST_FLAGS := -fPIC -pthread
CXX_CUDA_FLAGS := -arch=$(CUDA_ARCH) -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr
CXX_ROCM_FLAGS := --gcc-toolchain=$(shell $(CXX) -v 2>&1 | grep Configured | xargs -n1 echo | grep '^--prefix' | cut -d= -f2-) --gcc-triple=$(shell $(CXX) -dumpmachine) --offload-arch=$(ROCM_ARCH)
CXX_SYCL_CPU_FLAGS := --gcc-toolchain=$(shell $(CXX) -v 2>&1 | grep Configured | xargs -n1 echo | grep '^--prefix' | cut -d= -f2-) --gcc-triple=$(shell $(CXX) -dumpmachine) -fsycl -fsycl-targets=$(SYCL_GPU_ARCH) -I$(ONEAPI_BASE)/compiler/latest/include -I$(ONEAPI_BASE)/dpl/latest/include
CXX_SYCL_GPU_FLAGS := --gcc-toolchain=$(shell $(CXX) -v 2>&1 | grep Configured | xargs -n1 echo | grep '^--prefix' | cut -d= -f2-) --gcc-triple=$(shell $(CXX) -dumpmachine) -fsycl -fsycl-targets=spir64_x86_64 -I$(ONEAPI_BASE)/compiler/latest/include -I$(ONEAPI_BASE)/dpl/latest/include


test_cpu: test.cc
	$(CXX) -x c++ $(CXXFLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $< -L/usr/lib/x86_64-linux-gnu/openblas-serial -lopenblas -o $@

test_cuda: test.cc
	$(NVCC) -x cu $(CXXFLAGS) $(CXX_CUDA_FLAGS) -Xcompiler '$(CXX_HOST_FLAGS)' -DALPAKA_ACC_GPU_CUDA_ENABLED $< -lcublas -o $@
