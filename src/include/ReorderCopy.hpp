#pragma once 
#include <thrust/inner_product.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/iterator/permutation_iterator.h>

template <typename idx_type, typename val_type, idx_type dim>
struct cal_reorder_idx {

	typedef std::array<idx_type,dim> int_array;

	int_array order, n_dim;

	cal_reorder_idx(int_array n_dim,int_array order) 
	: order(order), n_dim(n_dim) {}

	// transform i to i'.
	idx_type operator()(int idx_s) {

		int_array idx_m;
		for (int i=0; i<dim; i++) {
			
			idx_type imod = 1;
			for (int j=0; j<i+1;j++) { imod *= n_dim[j]; }
			
			idx_m[i] = (idx_s % imod) * n_dim[i] / imod;
		}

		// reorder the multi-indices
		auto pitor = thrust::make_permutation_iterator(
	                  n_dim.begin(),order.begin());
		auto idx_pitor = thrust::make_permutation_iterator(
	                  idx_m.begin(),order.begin());

		int_array shift;	
		thrust::exclusive_scan(pitor,pitor+dim,shift.begin(),1,
		thrust::multiplies<std::size_t>());

		return thrust::inner_product(idx_pitor,idx_pitor+dim,
	                              shift.begin(),0);
	}
};


template <typename idx_type, typename val_type, idx_type dim>
class ReorderCopy {

	typedef std::array<idx_type,dim> int_array;

	int_array new_order, n_dim;

public:
	ReorderCopy(int_array n_dim, int_array new_order) 
	  : new_order(new_order),n_dim(n_dim) { }

	template <typename in_itor_t, typename out_itor_t>
	void operator()(in_itor_t in_begin, 
	                in_itor_t in_end, 
	                out_itor_t out_begin) {
		
		auto titor_begin = thrust::make_transform_iterator(
		                thrust::make_counting_iterator(0), 
		                cal_reorder_idx<idx_type,val_type,dim>
		                (n_dim,new_order));
	

	}

};
