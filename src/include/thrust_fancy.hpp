#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <iterator>

namespace quakins {

template <typename itor_type, typename phase_itor_type>
void evolve_with_phase(itor_type itor_begin, itor_type itor_end,
                       phase_itor_type phase_begin) {

  typedef typename std::iterator_traits<itor_type>::value_type val_type;
  typedef typename std::iterator_traits<phase_itor_type>::value_type phase_val_type;

  auto pexp = []__host__ __device__ (val_type val, const phase_val_type& phase) {
    val_type next_val;
    next_val.x =  val.x*cos(phase) - val.y*sin(phase);
    next_val.y =  val.x*sin(phase) + val.y*cos(phase);
    return next_val;
  };

  thrust::transform(thrust::device, itor_begin,itor_end, phase_begin, itor_begin, pexp);

}


} // namespace quakins
