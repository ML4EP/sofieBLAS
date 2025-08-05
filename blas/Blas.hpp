#pragma once

namespace traits
{

    template<typename TTag>
    class Blas;

} // namespace traits

template<typename TTag>
using Blas = traits::Blas<TTag>::Impl;
