#ifndef _PHASE_SPACE_INITIALIZATION_HPP_
#define _PHASE_SPACE_INITIALIZATION_HPP_

#include <fstream>
#include <cmath>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>

#include "details/initial_shapes.hpp"

template <typename idx_type, 
          typename val_type, 
          idx_type dim, typename ShapeFunctor>
struct Idx2Value {

  typedef Parameters<idx_type,val_type,dim> Parameters;

  typedef std::array<idx_type,dim> idx_array;
  typedef std::array<val_type,dim> val_array;

  idx_array n_dim, n_bd;
  val_array low_bound, h;
  
  ShapeFunctor *shape;

  __host__ __device__ 
  Idx2Value(idx_array n_dim, idx_array n_bd, 
            val_array lb, val_array h, ShapeFunctor *shape)
   : n_dim(n_dim),low_bound(lb),h(h), n_bd(n_bd), shape(shape) {}

  __host__ __device__ 
  val_type operator()(const idx_type& idx) {

    idx_array idx_m;
    for (int i=0; i<dim; i++) {
      
      idx_type imod = 1;
      for (int j=0; j<i+1;j++) { imod *= n_dim[j]; }

      idx_m[i] = (idx % imod) * n_dim[i] / imod;

    }
    val_array z;
    for (int i=0; i<dim; i++) 
      z[i] = low_bound[i]+.5*h[i]+h[i]*(idx_m[i]-n_bd[i]);

    return shape->write(z);  

  }

};

namespace quakins {


template <typename idx_type, 
          typename val_type, 
          idx_type dim, 
          template<typename W,typename,W,typename> typename ShapeFunctorTemplate>
class PhaseSpaceInitialization {

  typedef Parameters<idx_type,val_type,dim> Parameters;
  typedef ShapeFunctorTemplate<idx_type,val_type,dim,Parameters> ShapeFunctor;

  Parameters *p;
  ShapeFunctor *shape;

public:
  PhaseSpaceInitialization(Parameters* p) 
  : p(p) { shape = new ShapeFunctor(p);  }


  template <typename itor_type, typename ExecutionPolicy>
  __host__ 
  void operator()(const ExecutionPolicy & exec,
                  itor_type itor_begin, idx_type num, idx_type id) {

    idx_type n_shift = id*p->n_1d_per_dev - id*2*p->n_ghost[dim-1]
                       *p->n_1d_per_dev/p->n_all_local[dim-1];
    thrust::transform(exec, 
                      thrust::make_counting_iterator(n_shift),
                      thrust::make_counting_iterator(num+n_shift), 
                      itor_begin,
                      Idx2Value(p->n_all,p->n_ghost, p->low_bound,p->interval, shape));
    // the template parameters are automatically deduced via the constructor
    
    // Idx2Value calculate the value of distribution function from 1d index,
    // once the mesh number (n), lower boundary (low_bound)  and mesh interval 
    // of each dimension are given.
  }
};


} // namespace quakins


#endif /* _PHASE_SPACE_INITIALIZATION_HPP_ */


