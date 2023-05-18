#ifndef _PHASE_SPACE_INITIALIZATION_HPP_
#define _PHASE_SPACE_INITIALIZATION_HPP_

#include <fstream>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include "util.hpp"
#include "initialization.hpp"

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
	__host__ __device__
	void operator()(const ExecutionPolicy & exec,
	                itor_type itor_begin, idx_type num, idx_type start) {

struct ShapePolicy {

	typedef std::array<idx_type,dim> idx_array;
	typedef std::array<val_type,dim> val_array;

	idx_array n_dim;
	val_array low_bound, h;

	__host__ __device__ 
	ShapePolicy(idx_array n_dim, val_array lb, val_array h)
	 : n_dim(n_dim),low_bound(lb),h(h) {
	}
	__host__ __device__ 
	val_type operator()(const idx_type& idx) {

			printf("1");
		idx_array idx_m;
		for (int i=0; i<dim; i++) {
			
			idx_type imod = 1;
			for (int j=0; j<i+1;j++) { imod *= n_dim[j];}

			idx_m[i] = (idx % imod) / imod * n_dim[i];
			printf("%d",i);
		}

		return   std::exp(-std::pow(low_bound[0]+h[0]*idx_m[0]-5,2))
		       * std::exp(-std::pow(low_bound[1]+h[1]*idx_m[1]-5,2))
		       * std::exp(-std::pow(low_bound[2]+h[2]*idx_m[2],2))
		       * std::exp(-std::pow(low_bound[3]+h[3]*idx_m[3],2));

	}

};

		thrust::transform(exec, 
		                  thrust::make_counting_iterator(start),
		                  thrust::make_counting_iterator(num+start), 
		                  itor_begin,
		                  ShapePolicy(p->n,p->low_bound,p->interval));

	}
};


} // namespace quakins


#endif /* _PHASE_SPACE_INITIALIZATION_HPP_ */


