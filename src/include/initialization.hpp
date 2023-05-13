#ifndef _INITIALIZATION_HPP_
#define _INITIALIZATION_HPP_

#include <limits>
#include <omp.h>
#include "util.hpp"


template <typename idx_type, typename val_type, idx_type dim>
struct Parameters {

	idx_type n_dev;

	// # mesh grid
	idx_type n[dim];

	// # ghost mesh
	idx_type n_ghost[dim];

	// # total in each direction (n_tot=n1+2*n_ghost)
	idx_type n_tot[dim];

	// # n_tot 
	idx_type n_1d_tot, n_1d_per_dev;

	// boundary of the phase space (x1min,x2min,v1min,...,v2max)
	val_type bound[2*dim];

	// length of each direction in phase space
	val_type length[dim];

};

namespace quakins {

template<typename idx_type, typename val_type, idx_type dim>
void initDomain(Parameters<idx_type,val_type, dim>* p, 
                const std::map<std::string,std::string> &dmap) {

	assign(p->n[0], "nx1", dmap);
	assign(p->n[1], "nx2", dmap);
	assign(p->n[2], "nv1", dmap);
	assign(p->n[3], "nv2", dmap);
	assign(p->n_ghost[0], "nx1_ghost", dmap);
	assign(p->n_ghost[1], "nx2_ghost", dmap);
	assign(p->n_ghost[2], "nv1_ghost", dmap);
	assign(p->n_ghost[3], "nv2_ghost", dmap);

	assign(p->bound[0], "x1min", dmap);
	assign(p->bound[1], "x2min", dmap);
	assign(p->bound[2], "v1min", dmap);
	assign(p->bound[3], "v2min", dmap);
	assign(p->bound[4], "x1max", dmap);
	assign(p->bound[5], "x2max", dmap);
	assign(p->bound[6], "v1max", dmap);
	assign(p->bound[7], "v2max", dmap);
	
	std::transform(p->n, p->n+dim, p->n_ghost, p->n_tot,
	               [](idx_type a, idx_type b){ return a+2*b; });
	std::transform(p->bound, p->bound+dim, p->bound+dim, p->length,
	               [](val_type a, val_type b){ return b-a; });

	p->n_1d_tot = std::accumulate(p->n_tot,p->n_tot+dim,
	              static_cast<idx_type>(1),std::multiplies<idx_type>());

	std::copy(p->n_tot,p->n_tot+dim,std::ostream_iterator<idx_type>(std::cout," "));
	std::cout << p->n_1d_tot << std::endl;

	int temp; // todo: this is not very elegant.
	cudaGetDeviceCount(&temp);
	p->n_dev = static_cast<idx_type>(temp);

	p->n_1d_per_dev = p->n_1d_tot / p->n_dev;

	std::cout << "You have " << omp_get_num_procs() << " CPU hosts." << std::endl;
	std::cout << p->n_dev << " GPU devices are found: " << std::endl;
	for (int i=0; i<p->n_dev; i++) {
		cudaDeviceProp dev_prop;
		cudaGetDeviceProperties(&dev_prop,i);
		std::cout << "	" << i << ": " << dev_prop.name << std::endl;
	} // display the names of GPU devices
	
	std::cout << "Maxium of your int type: " 
	          << std::numeric_limits<idx_type>::max() << std::endl;

	std::size_t mem_size = p->n_1d_tot*sizeof(val_type)/1048576; 
	std::cout << "The Wigner function costs " 
	          << mem_size << "Mb of Memory, " 
	          << mem_size/p->n_dev << "Mb per GPU." << std::endl;


}

} // namespace quakins

#endif /* _INITIALIZATION_HPP_ */
