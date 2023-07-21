#pragma once

#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/adjacent_difference.h>

#include <fstream>


namespace quakins {

namespace details {


template <typename idx_type,
          typename val_type,
          idx_type dim,
          idx_type xdim, idx_type vdim>
class FluxBalanceCoordSpace {

  typedef thrust::tuple<val_type,val_type> TwoValTuple;
  typedef thrust::tuple<val_type,val_type,
                        val_type> ThreeValTuple;

  typedef thrust::tuple<val_type,val_type,
                        val_type,val_type> FourValTuple;

  const idx_type n_bd, nx, nv, n_all, n_chunk;
  const val_type dx;

  // flux function and shift length of each grid
  thrust::host_vector<val_type> alpha;
  thrust::device_vector<val_type> flux;

public:
  template <typename Parameters>
  FluxBalanceCoordSpace(Parameters *p, val_type dt) :
  n_bd(p->n_ghost[xdim]), nx(p->n[xdim]), 
  n_chunk(p->n_1d_per_dev/p->n_all_local[vdim]),
  nv(p->n_all_local[vdim]), n_all(p->n_1d_per_dev), dx(p->interval[xdim]) {
    
    val_type dv = p->interval[vdim];
    val_type dx = p->interval[xdim];
    val_type v_min = p->low_bound[vdim];


    flux.resize(n_chunk);
    alpha.resize(nv);


    auto v_itor_begin = 
         thrust::make_transform_iterator(
         thrust::make_counting_iterator(static_cast<idx_type>(0)),
         [v_min,dv,dx,dt](idx_type idx) {
           return dt*(v_min + dv*.5 + static_cast<val_type>(dv*idx))/dx; 
         });

    thrust::copy(v_itor_begin,v_itor_begin+nv,alpha.begin());
  }

  template <typename itor_type, typename vitor_type>
  void advance(itor_type itor_begin, itor_type itor_end, 
               vitor_type v_begin, int gpu) {
    // the outermost dimension is calculated sequentially

    val_type a, shift_f;
    int shift;

    auto itor_begin_l = itor_begin;

    using namespace thrust::placeholders;
    for (std::size_t i = 0; i<nv/2; i++) {

      a = std::modf(alpha[i],&shift_f);
      shift = -static_cast<int>(shift_f);

      auto zitor_neg_begin 
            = make_zip_iterator(thrust::make_tuple(
                      itor_begin_l+shift,
                      itor_begin_l+shift+1,  
                      itor_begin_l+shift+2));
      
      // calculate the flux function \flux
      thrust::transform(thrust::device, 
                        zitor_neg_begin+n_bd-1,
                        zitor_neg_begin-n_bd+n_chunk,
                        flux.begin()+n_bd-1,
      [a]__host__ __device__(ThreeValTuple tu){
        return a*(thrust::get<1>(tu) 
         -(1-a)*(1+a)/6.*(thrust::get<2>(tu)-thrust::get<1>(tu))
         -(2+a)*(1+a)/6.*(thrust::get<1>(tu)-thrust::get<0>(tu)));
      });

      for (int k=1; k<=shift; ++k) {
        
        thrust::transform(thrust::device,
                          itor_begin_l+n_bd-1+k,
                          itor_begin_l-n_bd+k+n_chunk,
                          flux.begin()+n_bd-1,
                          flux.begin()+n_bd-1, _2-_1);
      }

      thrust::adjacent_difference(thrust::device,
                                  flux.begin(),flux.end(),
                                  flux.begin());

      // calculate f[i](t+dt)=f[i](t) + flux[i-1/2] -flux[i+1/2]
      thrust::transform(thrust::device,
                        flux.begin(),flux.end(),
                        itor_begin_l,itor_begin_l, _2-_1);

      itor_begin_l += n_chunk;
    } // v < 0
      

    for (std::size_t i = nv/2; i<nv; i++) {
      
      a = std::modf(alpha[i],&shift_f);
      shift = static_cast<int>(-shift_f);

      auto zitor_pos_begin 
           = make_zip_iterator(thrust::make_tuple(
                               itor_begin_l+shift-1,
                               itor_begin_l+shift,  
                               itor_begin_l+shift+1));

      thrust::transform(thrust::device, 
                        zitor_pos_begin+n_bd-1,
                        zitor_pos_begin-n_bd+n_chunk,
                        flux.begin()+n_bd-1,
      [a]__host__ __device__(ThreeValTuple tu){
        return a*(thrust::get<1>(tu) 
         +(1-a)*(2-a)/6.*(thrust::get<2>(tu)-thrust::get<1>(tu))
         +(1-a)*(1+a)/6.*(thrust::get<1>(tu)-thrust::get<0>(tu)));
      });

      for (int k=-1; k>=shift; --k) {

        thrust::transform(thrust::device,
                          itor_begin_l+n_bd-1+k+1,
                          itor_begin_l-n_bd+k+n_chunk+1,
                          flux.begin()+n_bd-1,
                          flux.begin()+n_bd-1, _1+_2);
      }

      thrust::adjacent_difference(thrust::device,
                                  flux.begin(),flux.end(),
                                  flux.begin());

      // calculate f[i](t+dt)=f[i](t) + flux[i-1/2] -flux[i+1/2]
      thrust::transform(thrust::device,
                        flux.begin(),flux.end(),
                        itor_begin_l,itor_begin_l, _2-_1);

      itor_begin_l += n_chunk;
    } // v > 0


  }


};


} // namespace details
} // namespace quakins
