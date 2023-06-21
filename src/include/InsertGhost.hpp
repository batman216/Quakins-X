#pragma once 

#include <thrust/merge.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace quakins {


template <typename idx_type, typename itor_type>
void insertGhost(itor_type itor_begin,  idx_type n, idx_type n_ghost,
                 itor_type result,      idx_type n_tot) {

    auto key_ghost =
    thrust::make_transform_iterator(thrust::make_counting_iterator(
                                    static_cast<idx_type>(0)),
                                    [n_ghost]__host__ __device__(idx_type idx) 
                                    { return idx/n_ghost + (idx+n_ghost)/(2*n_ghost); });

    auto zeros = thrust::make_constant_iterator(0.0);

    auto key = 
    thrust::make_transform_iterator(thrust::make_counting_iterator(
                                    static_cast<idx_type>(0)),
                                    [n]__host__ __device__(idx_type idx)
                                    { return 1+3*(idx/n); });

    thrust::merge_by_key(key, key+n_tot*n,
                         key_ghost, key_ghost+2*n_tot*n_ghost,
                         itor_begin, zeros,
                         thrust::make_discard_iterator(),
                         result);

  }



} // namespace quakins
