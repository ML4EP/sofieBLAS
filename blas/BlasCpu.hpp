#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#    include "Blas.hpp"

#    include <alpaka/alpaka.hpp>

#    include <cblas.h>

class BlasCpu
{
public:
    BlasCpu(alpaka::QueueCpuBlocking& queue [[maybe_unused]])
    {
    }

    template<typename T, typename TIdx>
    inline void gemm(
        alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const& A,
        alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx> const& B,
        alpaka::BufCpu<T, alpaka::DimInt<1u>, TIdx>& C,
        TIdx size)
    {
        assert(alpaka::getExtentProduct(A) == size * size);
        assert(alpaka::getExtentProduct(B) == size * size);
        assert(alpaka::getExtentProduct(C) == size * size);

        // Set alpha and beta for GEMM: C = alpha*A*B + beta*C
        const T alpha = 1;
        const T beta = 0;

        // Perform C = A × B using OpenBLAS (column-major order)
        cblas_sgemm(
            CblasColMajor, // Layout
            CblasNoTrans, // Transpose A? No
            CblasNoTrans, // Transpose B? No
            size, // m, n, k
            size,
            size,
            alpha, // alpha
            A.data(), // A and its leading dimension
            size,
            B.data(), // B and its leading dimension
            size,
            beta, // beta
            C.data(), // C and its leading dimension
            size);
    }
};

namespace traits
{

    template<>
    class Blas<alpaka::TagCpuSerial>
    {
    public:
        using Impl = BlasCpu;
    };

    template<>
    class Blas<alpaka::TagCpuOmp2Blocks>
    {
    public:
        using Impl = BlasCpu;
    };

    template<>
    class Blas<alpaka::TagCpuOmp2Threads>
    {
    public:
        using Impl = BlasCpu;
    };

    template<>
    class Blas<alpaka::TagCpuTbbBlocks>
    {
    public:
        using Impl = BlasCpu;
    };

    template<>
    class Blas<alpaka::TagCpuThreads>
    {
    public:
        using Impl = BlasCpu;
    };

} // namespace traits

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
