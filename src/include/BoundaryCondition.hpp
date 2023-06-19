#pragma once

#include "util.hpp"

#include <thrust/iterator/reverse_iterator.h>


template<typename itor_type, typename idx_type> 
void bufferClean(itor_type itor_begin, itor_type itor_end,
                 idx_type nx, idx_type n_bd) {

   thrust::fill(itor_begin,itor_begin+n_bd,0.);
   thrust::fill(itor_end-n_bd,itor_end,0.);

   strided_chunk_range<itor_type,Flip> 
      mid_bd(itor_begin+n_bd+nx,itor_end-n_bd-nx,nx+2*n_bd, 2*n_bd);
  
   thrust::fill(mid_bd.begin(),mid_bd.end(),0.);
}

template <typename idx_type>
struct ReflectingBoundary {

  const idx_type nx, n_bd, n_chunk, n_step;

  ReflectingBoundary(idx_type nx, idx_type n_bd, 
                     idx_type n_step, idx_type n_chunk)
  : nx(nx), n_bd(n_bd), n_chunk(n_chunk), n_step(n_step){}


  template<typename itor_type> 
  void implement(itor_type itor_begin, itor_type itor_end, char flag) {

    bufferClean(itor_begin,itor_end,nx,n_bd);

    itor_type itor_neg_begin = itor_begin, 
              itor_pos_begin = itor_end-n_chunk;
    for (int i=0; i<n_step/2; i++) {
      
      strided_chunk_range<itor_type,Flip>
        right_inside(itor_pos_begin+nx, itor_pos_begin+n_chunk, 2*n_bd+nx,n_bd);
      strided_chunk_range<itor_type>
        right_bd(itor_neg_begin+nx+n_bd, itor_neg_begin+n_chunk, 2*n_bd+nx,n_bd);

      thrust::copy(right_inside.begin(),right_inside.end(),right_bd.begin());

      strided_chunk_range<itor_type,Flip>
        left_inside(itor_neg_begin+n_bd, itor_neg_begin+n_chunk-nx, 2*n_bd+nx,n_bd);
      strided_chunk_range<itor_type>
        left_bd(itor_pos_begin, itor_pos_begin+n_chunk-nx-n_bd, 2*n_bd+nx,n_bd);

      thrust::copy(left_inside.begin(),left_inside.end(),left_bd.begin());


      itor_neg_begin += n_chunk;
      itor_pos_begin -= n_chunk;

    }


  }

};


template <typename idx_type>
struct PeriodicBoundary {


  const idx_type nx, n_bd, n_chunk, n_step;

  PeriodicBoundary(idx_type nx, idx_type n_bd, 
                   idx_type n_step, idx_type n_chunk)
  : nx(nx), n_bd(n_bd), n_chunk(n_chunk), n_step(n_step){}

  template<typename itor_type> __host__
  void implement(itor_type itor_begin, itor_type itor_end,char flag) {

    strided_chunk_range<itor_type> 
      left_bd(itor_begin,itor_end-n_bd,nx+2*n_bd, n_bd);
    strided_chunk_range<itor_type> 
      right_bd(itor_begin+nx+n_bd,itor_end-n_bd,nx+2*n_bd, n_bd);

    strided_chunk_range<itor_type> 
      left_inside(itor_begin+n_bd,itor_end-n_bd,nx+2*n_bd, n_bd);
    strided_chunk_range<itor_type> 
      right_inside(itor_begin+nx,itor_end-n_bd,nx+2*n_bd, n_bd);

    thrust::copy(thrust::device,
                 left_inside.begin(),left_inside.end(),
                 right_bd.begin());
    thrust::copy(thrust::device,
                 right_inside.begin(),right_inside.end(),
                 left_bd.begin());
  }


};

namespace quakins {

template <typename idx_type, 
          template<typename> typename Policy>
class BoundaryCondition {
  Policy<idx_type> *policy;

public:

  BoundaryCondition(idx_type nx, idx_type n_bd, 
                    idx_type n_step, idx_type n_chunk) { 
    policy = new Policy<idx_type>(nx,n_bd,n_step,n_chunk);
  }


  template<typename itor_type> __host__
  void operator()(itor_type itor_begin, itor_type itor_end,char flag) {

    policy->implement(itor_begin,itor_end,flag);
  }


};



} // namespace quakins
