# sofieBLAS
sofieBLAS is an abstract C++ (header-only) interface for BLAS operations targeting heterogeneous architectures.  It currently supports only ALPAKA buffers and the GEMM operation, acting as a thin, efficient wrapper over existing BLAS libraries such as OpenBLAS, MKL, cuBLAS, and others- allowing the actual backend to be selected through template-based dispatching using traits.

We plan to extend support to more BLAS routines and buffer types in future releases.

## Features

- Unified Interface: Common C++ API over multiple BLAS backends.
- Heterogeneous Support: CPU (OpenBLAS, MKL) and GPU (cuBLAS) support.
- Template-Based Dispatching: Backend selection via traits at compile-time.
- Header-Only: Lightweight, easy to integrate- no separate compilation required.
- Minimal Dependency Overhead: Only depends on the backend BLAS libraries of choice.

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

  // Create sofieBLAS instance for CPU backend
  sofieBLAS<alpaka::TagCpuSerial> blas(queue);

  // Perform GEMM: C = alpha * A * B + beta * C
  blas.gemm('n', 'n', size, size, size, 1.0f, A, size, B, size, 0.0f, C, size);

  alpaka::wait(queue);
  std::cout << "GEMM completed on CPU backend.\n";

  return 0;
}
```


## Contributing

Contributions, feature suggestions, and issue reports are welcome! Feel free to [open a PR](https://github.com/ML4EP/sofieBLAS/pulls) or an [issue](https://github.com/ML4EP/sofieBLAS/issues).



## License

[GNU General Public License v3.0](https://github.com/ML4EP/sofieBLAS/blob/main/LICENSE)

