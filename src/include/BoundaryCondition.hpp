#pragma once

#include "util.hpp"

#include <thrust/iterator/reverse_iterator.h>


template<typename itor_type, typename idx_type> __host__
void bufferClean(itor_type itor_begin, itor_type itor_end,
                 idx_type nx, idx_type n_bd) {
   strided_chunk_range<itor_type> 
      right_bd(itor_begin+nx+n_bd,itor_end,nx+2*n_bd, n_bd);
   strided_chunk_range<itor_type> 
      left_bd(itor_begin,itor_end,nx+2*n_bd, n_bd);
  
   thrust::fill(right_bd.begin(),right_bd.end(),0);
   thrust::fill(left_bd.begin(),left_bd.end(),0);

}

struct ReflectingBoundary {

  template<typename itor_type,typename idx_type> __host__
  static void implement(itor_type itor_begin, itor_type itor_end,
                        idx_type nx, idx_type n_bd, char flag) {
 
    auto half = (itor_end-itor_begin)/2;
    using ritor_type = thrust::reverse_iterator<itor_type>;
    ritor_type ritor_begin(itor_end);
    
    strided_chunk_range<ritor_type> 
      right_inside(ritor_begin+n_bd,ritor_begin+half,nx+2*n_bd, n_bd);
    strided_chunk_range<itor_type> 
      right_bd(itor_begin+n_bd+nx,itor_begin+half,nx+2*n_bd, n_bd);
    // 反着拿，顺着放
    thrust::copy(thrust::device,
                 right_inside.begin(),right_inside.end(),
                 right_bd.begin());

    strided_chunk_range<itor_type> 
      left_inside(itor_begin+n_bd,itor_begin+half,nx+2*n_bd, n_bd);
    strided_chunk_range<ritor_type> 
      left_bd(ritor_begin+nx+n_bd,ritor_begin+half,nx+2*n_bd, n_bd);
    // 顺着拿，反着放
    thrust::copy(thrust::device,
                 left_inside.begin(),left_inside.end(),
                 left_bd.begin());

  }


};




struct PeriodicBoundaryPara {

  template<typename itor_type,typename idx_type> __host__
  static void implement(itor_type itor_begin, itor_type itor_end,
                        idx_type nx, idx_type n_bd, char flag) {

    strided_chunk_range<itor_type> 
      left_bd(itor_begin,itor_end-n_bd,nx+2*n_bd, n_bd);
    strided_chunk_range<itor_type> 
      right_bd(itor_begin+nx+n_bd,itor_end-n_bd,nx+2*n_bd, n_bd);

    strided_chunk_range<itor_type> 
      left_inside(itor_begin+n_bd,itor_end-n_bd,nx+2*n_bd, n_bd);
    strided_chunk_range<itor_type> 
      right_inside(itor_begin+nx,itor_end-n_bd,nx+2*n_bd, n_bd);

    if (flag=='l')
      thrust::copy(thrust::device,
                 left_inside.begin(),left_inside.end(),
                 right_bd.begin());
    if (flag=='r')
      thrust::copy(thrust::device,
                 right_inside.begin(),right_inside.end(),
                 left_bd.begin());
  }


};


struct PeriodicBoundary {

  template<typename itor_type,typename idx_type> __host__
  static void implement(itor_type itor_begin, itor_type itor_end,
                        idx_type nx, idx_type n_bd, char flag) {

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

template <typename idx_type, typename Policy>
class BoundaryCondition {

protected:
  const idx_type nx, n_bd;

public:

  BoundaryCondition(idx_type nx, idx_type n_bd) 
  : nx(nx), n_bd(n_bd){}


  template<typename itor_type> __host__
  void operator()(itor_type itor_begin, itor_type itor_end,char flag) {

    Policy::implement(itor_begin,itor_end,nx,n_bd,flag);
  }


};



} // namespace quakins
