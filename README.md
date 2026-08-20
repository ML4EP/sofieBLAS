# sofieBLAS
sofieBLAS is an abstract C++ (header-only) interface for BLAS operations targeting heterogeneous architectures. It currently supports only ALPAKA buffers and the GEMM operation, acting as a thin, efficient wrapper over existing BLAS libraries such as OpenBLAS, MKL, BLIS, Apple Accelerate, cuBLASLt, and hipBLASLt - allowing the actual backend to be selected through template-based dispatching using traits.

We plan to extend support to more BLAS routines and buffer types in future releases.

## Features

- Unified Interface: Common C++ API over multiple BLAS backends.
- Heterogeneous Support:
  - CPU: OpenBLAS, MKL, BLIS, Apple Accelerate (any CBLAS-compatible library).
  - GPU: NVIDIA (cuBLASLt) and AMD (hipBLASLt).
- Template-Based Dispatching: Backend selection via traits at compile-time, keyed on the Alpaka accelerator tag (e.g. `alpaka::TagCpuSerial`, `alpaka::TagGpuCudaRt`, `alpaka::TagGpuHipRt`).
- Header-Only: Lightweight, easy to integrate- no separate compilation required.
- Minimal Dependency Overhead: Only depends on the backend BLAS libraries of choice.
- One File Per Backend: Each vendor library (CPU or GPU) lives in its own header under `include/sofieBLAS/backends/<cpu|cuda|hip>/`, selected at compile time via a preprocessor macro (`ALPAKA_ACC_GPU_CUDA_ENABLED`, `ALPAKA_ACC_GPU_HIP_ENABLED`, `SOFIEBLAS_USE_OPENBLAS`, `SOFIEBLAS_USE_MKL`, `SOFIEBLAS_USE_BLIS`, `SOFIEBLAS_USE_ACCELERATE`) so only the code for the library you actually build against gets compiled.

## Selecting a backend

Backend selection happens entirely at compile time, in two steps:

1. **Compiler-flag macros decide the backend implementations.** A macro (or macro pair) must be defined to compile in a given backend's header - see the table below. For CPU, if `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` is defined but none of the `SOFIEBLAS_USE_*` macros are, sofieBLAS defaults to OpenBLAS; the GPU backends have no such default (each is tied 1:1 to its `ALPAKA_ACC_GPU_*_ENABLED` macro, so there's nothing to default between).
2. **The Alpaka tag you instantiate `sofieBLAS<Tag>` with decides which compiled-in backend a given call site actually uses**, via the `traits::sofieBLAS<Tag>` specialization (e.g. `sofieBLAS<alpaka::TagGpuHipRt>` resolves to the hipBLASLt backend). There is no runtime dispatch - if the macro for that tag's backend wasn't defined, the code simply won't compile.

| Accelerator | Macro(s) | Alpaka tag |
| --- | --- | --- |
| CPU (OpenBLAS) | `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` + `SOFIEBLAS_USE_OPENBLAS` | `alpaka::TagCpuSerial` (or other `TagCpu*`) |
| CPU (Intel MKL) | `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` + `SOFIEBLAS_USE_MKL` | `alpaka::TagCpuSerial` (or other `TagCpu*`) |
| CPU (BLIS) | `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` + `SOFIEBLAS_USE_BLIS` | `alpaka::TagCpuSerial` (or other `TagCpu*`) |
| CPU (Apple Accelerate) | `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` + `SOFIEBLAS_USE_ACCELERATE` | `alpaka::TagCpuSerial` (or other `TagCpu*`) |
| NVIDIA GPU (cuBLASLt) | `ALPAKA_ACC_GPU_CUDA_ENABLED` | `alpaka::TagGpuCudaRt` |
| AMD GPU (hipBLASLt) | `ALPAKA_ACC_GPU_HIP_ENABLED` | `alpaka::TagGpuHipRt` |

`sofieBLAS/sofieBLAS.hpp` includes the matching backend header(s) for you based on these macros; nothing else needs to change in your source beyond picking the right tag.

## Building tests and benchmarks

sofieBLAS itself is header-only (`add_subdirectory`/`find_package(sofieBLAS)` gives you an `INTERFACE` target with no build step). Tests and benchmarks are opt-in via CMake options and auto-detect whichever backends are available on the machine for each backend target (`test_cpu`/`test_cuda`/`test_hip`, `bench_cpu`/`bench_cuda`/`bench_hip`) and is skipped if its dependency isn't found:

```bash
cmake -B build -S . -DSOFIEBLAS_BUILD_TESTS=ON -DSOFIEBLAS_BUILD_BENCHMARKS=ON
cmake --build build -j"$(nproc)"

# Run the correctness tests (matmul/gemm/gemmrelu/gemmgelu vs. a reference
# implementation, per available backend)
ctest --test-dir build --output-on-failure

# Run the GEMM throughput benchmark
./build/benchmark/bench_cuda -w 5 -n 20 --sizes 256,512,1024,2048,4096
```

`CPU_BLAS_LIB` (`OpenBLAS`/`MKL`/`BLIS`/`Accelerate`) and `CUDA_BASE`/`ROCM_BASE`/`ONEAPI_BASE`/`BLIS_BASE` are available as `-D` cache variables to point at non-default install locations; alpaka is picked up via `find_package(alpaka)` if already installed, otherwise fetched automatically. See `tests/CMakeLists.txt` and `benchmark/CMakeLists.txt` for the target definitions, and `cmake/SofieBLASBackends.cmake` for the shared backend-detection logic.

## Usage example

```cpp
#include "sofieBLAS/sofieBLAS.hpp"
#include <alpaka/alpaka.hpp>
#include <iostream>

int main() {
  constexpr uint32_t size = 4;

  // Create Alpaka CPU device and blocking queue
  alpaka::PlatformCpu platform;
  auto device = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevCpu, alpaka::Blocking> queue{device};

  // Allocate and initialize matrices A, B, C on CPU
  auto A = alpaka::allocBuf<float, uint32_t>(device, size * size);
  auto B = alpaka::allocBuf<float, uint32_t>(device, size * size);
  auto C = alpaka::allocBuf<float, uint32_t>(device, size * size);
  // (Initialize A and B here...)

  // Create sofieBLAS instance for the CPU backend selected via
  // SOFIEBLAS_USE_OPENBLAS / SOFIEBLAS_USE_MKL / SOFIEBLAS_USE_BLIS / SOFIEBLAS_USE_ACCELERATE
  sofieBLAS<alpaka::TagCpuSerial> blas(queue);

  // C = alpha * op(A) * op(B) + beta * C  (leading dimensions inferred from m, n, k)
  blas.matmul('N', 'N', size, size, size, 1.0f, A, B, 0.0f, C);

  alpaka::wait(queue);
  std::cout << "GEMM completed on CPU backend.\n";

  return 0;
}
```

Switching to a GPU is the same code shape, just a different Alpaka tag, queue/device type, and build-time macro. For example, on AMD (built with `-DALPAKA_ACC_GPU_HIP_ENABLED`):

```cpp
alpaka::PlatformHipRt platform;
auto device = alpaka::getDevByIdx(platform, 0u);
alpaka::Queue<alpaka::DevHipRt, alpaka::NonBlocking> queue{device};
sofieBLAS<alpaka::TagGpuHipRt> blas(queue);

blas.matmul('N', 'N', size, size, size, 1.0f, dA, dB, 0.0f, dC);
```

The GPU backends (`BlasCuda`, `BlasHip`) additionally expose `gemmrelu`/`gemmgelu` (fused bias + activation via cuBLASLt/hipBLASLt epilogues), `gemmStridedBatched`, and `addLayoutConfig` (declares a call site's shape ahead of the first call, which feeds the algorithm cache described below).

## Dynamic GEMM shapes and the algorithm cache

A GEMM call computes `C = alpha * op(A) * op(B) + beta * C`: A and B are the input matrices, C the output, and `op` an optional transpose. The CUDA backend (`BlasCuda`, over cuBLASLt) and the HIP backend (`BlasHip`, over hipBLASLt) implement the following identically. One instance serves GEMM calls at sizes that vary at runtime: for every call, each of the three matrices has its dimensions stamped into a persistent per-matrix layout descriptor right before the multiply, so no layout is ever tied to a fixed shape.

Every matmul also needs an algorithm: the concrete GEMM kernel and its configuration, chosen by querying the library's heuristic (`cublasLtMatmulAlgoGetHeuristic` / `hipblasLtMatmulAlgoGetHeuristic`) for the given transpose settings, epilogue, and matrix dimensions. That query runs on the host and is not free, so its results are cached, keyed by the same parameters. Left at that, a model with runtime-varying sizes would pay one heuristic query per distinct size it meets, and the cache would grow with every new size.

The envelope mechanism removes that cost. An envelope is the largest shape a GEMM call site will ever run, declared through `addLayoutConfig(m, n, k, lda, ldb, ldc, transa, transb)`, which a generated Session constructor calls once per call site with its own construction-time dimensions. `addLayoutConfig` stores the envelope and resolves its algorithm immediately. At matmul time the call is matched to a declared envelope (the contraction dimension must match exactly, which is what ties a call to its own call site), and the algorithm is looked up at the envelope's dimensions rather than the call's, so every runtime size a call site produces shares that one cache entry: no further heuristic queries, no cache growth. A call covered by no envelope is resolved and cached at its exact shape. The library may also reject an envelope's algorithm at a specific smaller size (m=1 was found to do this in testing); the backend checks for this and falls back to exact-shape resolution for that call.

The cache is unbounded by default. Pass a limit at construction to cap it with LRU eviction:

```cpp
sofieBLAS<alpaka::TagGpuCudaRt> blas(queue);       // unbounded cache (default)
sofieBLAS<alpaka::TagGpuCudaRt> capped(queue, 32); // at most 32 entries, LRU eviction
sofieBLAS<alpaka::TagGpuHipRt>  hipCapped(queue, 32);
```

`algoCacheSize()` returns the current entry count and `layoutStats()` returns counters (`heuristicQueries`, `envelopeRejects`, `evictions`) for inspecting cache behaviour.

The exact-shape fallback uses `cublasLtMatmulAlgoCheck` on CUDA. hipBLASLt has no equivalent in its core API, so the HIP backend uses `hipblaslt_ext::matmulIsAlgoSupported` from `hipblaslt-ext.hpp` for the same check.


## Contributing

Contributions, feature suggestions, and issue reports are welcome! Feel free to [open a PR](https://github.com/ML4EP/sofieBLAS/pulls) or an [issue](https://github.com/ML4EP/sofieBLAS/issues).



## License

[GNU General Public License v3.0](https://github.com/ML4EP/sofieBLAS/blob/main/LICENSE)

