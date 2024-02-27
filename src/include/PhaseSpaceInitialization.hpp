/**
 * @file    PhaseSpaceInitialization.hpp
 * @author  Tian-Xing Hu 
 * @date    2023.10.13
 * @brief   
 */

#ifndef _PHASE_SPACE_INITIALIZATION_HPP_
#define _PHASE_SPACE_INITIALIZATION_HPP_

#include <fstream>
#include <cmath>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>

#include "initial_shapes.hpp"

namespace quakins {

template <typename idx_type, typename val_type, int dim_x, int dim_v>
struct Parameters;

template <typename idx_type, typename val_type, 
          int dim, template<typename,int> typename ShapeFunctor>
struct Idx2Value {

  typedef shape_packet_traits<val_type,dim,
                              ShapeFunctor>::packet ShapePacket;
 
  using idx_XV_t = std::array<idx_type,dim>;
  using val_XV_t = std::array<val_type,dim>;
   
  idx_XV_t n_dim, n_bd;
  val_XV_t low_bound, h;
  ShapePacket p;

  __host__ __device__ 
  Idx2Value(idx_XV_t n_dim, idx_XV_t n_bd, 
            val_XV_t lb, val_XV_t h, ShapePacket p)
   : n_dim(n_dim),low_bound(lb), h(h), n_bd(n_bd), p(p) {}

  __host__ __device__ 
  val_type operator()(const idx_type& idx) {

    idx_XV_t idx_m;
    for (int i=0; i<dim; i++) {
      
      idx_type imod = 1;
      for (int j=0; j<i+1;j++) { imod *= n_dim[j]; }

      idx_m[i] = (idx % imod) * n_dim[i] / imod;

    }
    val_XV_t z;
    for (int i=0; i<dim; i++) 
      z[i] = low_bound[i]+.5*h[i]+h[i]*(idx_m[i]-n_bd[i]);

    return ShapeFunctor<val_type,dim>::shape(z,p);  

  }

};

template <typename idx_type, typename val_type, 
          int dim_x, int dim_v, 
          template<typename,int> typename ShapeFunctor>
class PhaseSpaceInitialization {

  typedef shape_packet_traits<val_type,dim_x+dim_v,
                              ShapeFunctor>::packet ShapePacket;
  typedef Parameters<idx_type,val_type,dim_x,dim_v> Pmts;

  Pmts *p;
  ShapePacket *packet;

public:
  PhaseSpaceInitialization(Pmts *p) :p(p) {} 

  template <typename Container>
  void operator()(Container& storge) {

    using idx_XV_t = std::array<idx_type,dim_x+dim_v>;
    using val_XV_t = std::array<val_type,dim_x+dim_v>;
    using thrust::transform;
   
    idx_XV_t num,num_ghost;
    val_XV_t low_bound, interval;

    packet = new ShapePacket();

    for (int i=0; i<dim_v; ++i) {
      num[i] = p->n_all_v[i];
      num_ghost[i] = p->n_ghost_v[i];
      low_bound[i] = p->vmin[i];
      interval[i]  = p->dv[i];
    }

    for (int i=dim_v; i<dim_x+dim_v; ++i) {
      num[i] = p->n_all_x[i-dim_v];
      num_ghost[i] = p->n_ghost_x[i-dim_v];
      low_bound[i] = p->xmin[i-dim_v];
      interval[i]  = p->dx[i-dim_v];
    }

    idx_type n_shift = p->mpi_rank * p->n_whole_loc 
                     - p->mpi_rank * 2*num_ghost[dim_x+dim_v-1]
                     * std::accumulate(num.begin(),num.end()-1,1,
                                       std::multiplies<idx_type>()); 
    
    storge.resize(p->n_whole_loc);

    auto citor_begin = thrust::make_counting_iterator(n_shift);
    transform(citor_begin, citor_begin + p->n_whole_loc, storge.begin(),
              Idx2Value<idx_type,val_type,dim_x+dim_v,ShapeFunctor>
              (num, num_ghost, low_bound, interval, *packet));
    // the template parameters are automatically deduced via the constructor
    
    // Idx2Value calculate the value of distribution function from 1d index,
    // once the mesh number (n), lower boundary (low_bound)  and mesh interval 
    // of each dimension are given.
  }
};


} // namespace quakins


#endif /* _PHASE_SPACE_INITIALIZATION_HPP_ */


