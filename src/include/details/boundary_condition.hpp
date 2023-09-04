#pragma once 
#include "../util.hpp"

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

    idx_type n_tot = n_bd*2+nx;
    for (int i=0; i<n_step/2; i++) {
      
      strided_chunk_range<itor_type,Flip>
        right_inside(itor_pos_begin+nx, itor_pos_begin+n_chunk, n_tot,n_bd);
      strided_chunk_range<itor_type>
        right_bd(itor_neg_begin+nx+n_bd, itor_neg_begin+n_chunk, n_tot,n_bd);

      thrust::copy(right_inside.begin(),right_inside.end(),right_bd.begin());
      
      // For a cylindrical system, you'd better to only set r=0 boundary reflect,
      // and r=rmax boundary free
      strided_chunk_range<itor_type,Flip>
        left_inside(itor_neg_begin+n_bd, itor_neg_begin+n_chunk-nx, n_tot,n_bd);
      strided_chunk_range<itor_type>
        left_bd(itor_pos_begin, itor_pos_begin+n_chunk-nx-n_bd, n_tot,n_bd);

      thrust::copy(left_inside.begin(),left_inside.end(),left_bd.begin());

      // r=rmax free 
      /*
      strided_chunk_range<itor_type>
        right_inside_neg(itor_neg_begin+nx, itor_neg_begin+n_chunk, n_tot,n_bd);
      strided_chunk_range<itor_type>
        right_bd_neg(itor_neg_begin+nx+n_bd, itor_neg_begin+n_chunk, n_tot,n_bd);

      thrust::copy(right_inside_neg.begin(),right_inside_neg.end(),right_bd_neg.begin());
*/


      // padding the other side to avoid sharp edge at the boundary, 
      // which would introduce a numerical error.
      strided_chunk_range<itor_type>
        left_inside_neg(itor_neg_begin+n_bd, itor_neg_begin+n_chunk, n_tot,n_bd);
      strided_chunk_range<itor_type>
        left_bd_neg(itor_neg_begin, itor_neg_begin+n_chunk, n_tot, n_bd);

      thrust::copy(left_inside_neg.begin(),left_inside_neg.end(),left_bd_neg.begin());

      strided_chunk_range<itor_type>
        right_inside_pos(itor_pos_begin+nx, itor_pos_begin+n_chunk, n_tot,n_bd);
      strided_chunk_range<itor_type>
        right_bd_pos(itor_pos_begin+nx+n_bd, itor_pos_begin+n_chunk, n_tot,n_bd);

      thrust::copy(right_inside_pos.begin(),right_inside_pos.end(),right_bd_pos.begin());

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


