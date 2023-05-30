#ifndef _PHASE_SPACE_INITIALIZATION_HPP_
#define _PHASE_SPACE_INITIALIZATION_HPP_

#include <fstream>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>

template <typename idx_type, 
          typename val_type, 
          idx_type dim>
struct Idx2Value {

  typedef std::array<idx_type,dim> idx_array;
  typedef std::array<val_type,dim> val_array;

  idx_array n_dim, n_bd;
  val_array low_bound, h;

  __host__ __device__ 
  Idx2Value(idx_array n_dim, idx_array n_bd, val_array lb, val_array h)
   : n_dim(n_dim),low_bound(lb),h(h), n_bd(n_bd) {
  }

  __host__ __device__ 
  val_type operator()(const idx_type& idx) {

    idx_array idx_m;
    for (int i=0; i<dim; i++) {
      
      idx_type imod = 1;
      for (int j=0; j<i+1;j++) { imod *= n_dim[j]; }

      idx_m[i] = (idx % imod) * n_dim[i] / imod;

    }
    return   std::exp(-std::pow(low_bound[0]+h[0]*(idx_m[0]-n_bd[0]),2)/2)
           * std::exp(-std::pow(low_bound[1]+h[1]*(idx_m[1]-n_bd[1]),2))
           * std::exp(-std::pow(low_bound[2]+h[2]*(idx_m[2]-n_bd[2])-5,2))
           * std::exp(-std::pow(low_bound[3]+h[3]*(idx_m[3]-n_bd[3])-5,2));
           //* (1.+std::sin(M_PI*0.1*(low_bound[2]+h[2]*(idx_m[2]-n_bd[2])))*0.2);
  }

};


namespace quakins {


template <typename idx_type, 
          typename val_type, 
          idx_type dim>
class PhaseSpaceInitialization {

  Parameters<idx_type,val_type,dim> *p;

public:
  PhaseSpaceInitialization(Parameters<idx_type,val_type,dim>* p) 
  : p(p) {}

  template <typename itor_type, typename ExecutionPolicy>
  __host__ 
  void operator()(const ExecutionPolicy & exec,
                  itor_type itor_begin, idx_type num, idx_type start) {

    thrust::transform(exec, 
                      thrust::make_counting_iterator(start),
                      thrust::make_counting_iterator(num+start), 
                      itor_begin,
                      Idx2Value(p->n_tot,p->n_ghost, p->low_bound,p->interval));

    // Idx2Value calculate the value of distribution function from 1d index,
    // once the mesh number (n), lower boundary (low_bound)  and mesh interval 
    // of each dimension are given.
  }
};


} // namespace quakins


#endif /* _PHASE_SPACE_INITIALIZATION_HPP_ */


