#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include "Blas.hpp"

#    include <alpaka/alpaka.hpp>

#    include <cublas_v2.h>

// Check for cuBLAS errors
#    define CUBLAS_CHECK(err)                                                                                         \
        if(err != CUBLAS_STATUS_SUCCESS)                                                                              \
        {                                                                                                             \
            std::cerr << "cuBLAS Error at line " << __LINE__ << "\n";                                                 \
            exit(EXIT_FAILURE);                                                                                       \
        }

class BlasCuda
{
public:
    BlasCuda(alpaka::QueueCudaRtNonBlocking& queue) : m_queue{queue}
    {
        CUBLAS_CHECK(cublasCreate(&m_handle));
        CUBLAS_CHECK(cublasSetStream(m_handle, m_queue.getNativeHandle()));
    }

    ~BlasCuda()
    {
        CUBLAS_CHECK(cublasDestroy(m_handle));
    }

    template<typename T, typename TIdx>
    inline void gemm(
        alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const& A,
        alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const& B,
        alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx>& C,
        TIdx size)
    {
        assert(alpaka::getExtentProduct(A) == size * size);
        assert(alpaka::getExtentProduct(B) == size * size);
        assert(alpaka::getExtentProduct(C) == size * size);

        // Set alpha and beta for GEMM: C = alpha*A*B + beta*C
        const T alpha = 1;
        const T beta = 0;

        // Perform C = A × B using cuBLAS (column-major)
        CUBLAS_CHECK(cublasSgemm(
            m_handle,
            CUBLAS_OP_N, // Transpose A? No
            CUBLAS_OP_N, // Transpose B? No
            size, // m, n, k
            size,
            size,
            &alpha, // alpha
            A.data(), // A and its leading dimension
            size,
            B.data(), // B and its leading dimension
            size,
            &beta, // beta
            C.data(), // C and its leading dimension
            size));
    }

private:
    alpaka::QueueCudaRtNonBlocking m_queue;
    cublasHandle_t m_handle;
};

namespace traits
{

    template<>
    class Blas<alpaka::TagGpuCudaRt>
    {
    public:
        using Impl = BlasCuda;
    };

} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
