#pragma once

#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <array>

#include <iostream>

#include "Weighter.hpp"

namespace quakins {

template<typename val_type>
struct Integrator {
  
  const std::size_t n, n_batch;
  const val_type coeff;

  Integrator(std::size_t n, std::size_t n_b, val_type a, val_type b) :
    n(n), n_batch(n_b),
    coeff((b-a)/3./static_cast<val_type>(n)) {}

  template <typename itor_type>
  void operator()(itor_type in_begin, itor_type out_begin) {

    val_type C = this->coeff;
    auto zitor_begin = thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::make_counting_iterator(0),in_begin));

    typedef thrust::tuple<int,val_type> Tuple;
    auto titor_begin = make_transform_iterator(zitor_begin,
    [C]__host__ __device__(Tuple _tuple){ 

       return static_cast<val_type>(thrust::get<1>(_tuple) 
                       *(thrust::get<0>(_tuple)%2==0? 2.*C:4.*C)); 

    });
    
    std::size_t nn = this->n;
    auto binary_pred = [nn]__host__ __device__(int i,int j) { return i/nn==j/nn ; };

    thrust::reduce_by_key(thrust::device,
                          thrust::make_counting_iterator(0),  // input key
                          thrust::make_counting_iterator(static_cast<int>(n*n_batch)),
                          titor_begin,                        // input value
                          thrust::make_discard_iterator(),    // output key
                          out_begin,                          // output value
                          binary_pred);
  } // end of operator()

};



} // namespace quakins


