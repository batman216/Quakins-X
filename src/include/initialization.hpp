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
	
	std::string v_shape;
	// characteristic velocties
	val_type v1, v2, v3, v4;

};

namespace quakins {

template<typename idx_type, typename val_type, idx_type dim>
void init(Parameters<idx_type,val_type, dim>* p) {

	std::ifstream input_file("quakins.input");
	auto d_map = read_box(input_file, "domain");

	assign(p->n[0], "nx1", d_map);
	assign(p->n[1], "nx2", d_map);
	assign(p->n[2], "nv1", d_map);
	assign(p->n[3], "nv2", d_map);
	assign(p->n_ghost[0], "nx1_ghost", d_map);
	assign(p->n_ghost[1], "nx2_ghost", d_map);
	assign(p->n_ghost[2], "nv1_ghost", d_map);
	assign(p->n_ghost[3], "nv2_ghost", d_map);

	assign(p->bound[0], "x1min", d_map);
	assign(p->bound[1], "x2min", d_map);
	assign(p->bound[2], "v1min", d_map);
	assign(p->bound[3], "v2min", d_map);
	assign(p->bound[4], "x1max", d_map);
	assign(p->bound[5], "x2max", d_map);
	assign(p->bound[6], "v1max", d_map);
	assign(p->bound[7], "v2max", d_map);
	
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
	std::cout << "Maxium of your int type: " 
	          << std::numeric_limits<idx_type>::max() << std::endl;

	for (int i=0; i<p->n_dev; i++) {
		cudaDeviceProp dev_prop;
		cudaGetDeviceProperties(&dev_prop,i);
		std::cout << "	" << i << ": " << dev_prop.name << std::endl;
	} // display the names of GPU devices
	
	std::size_t mem_size = p->n_1d_tot*sizeof(val_type)/1048576; 
	std::cout << "The Wigner function costs " 
	          << mem_size << "Mb of Memory, " 
	          << mem_size/p->n_dev << "Mb per GPU." << std::endl;

	
	auto v_map = read_box(input_file,"initial");

	assign(p->v1,"v1",v_map);
	assign(p->v2,"v2",v_map);
	assign(p->v3,"v3",v_map);
	assign(p->v4,"v4",v_map);

	p->v_shape = v_map["shape"];
	std::cout << "initial shape of the Wigner function velocity space: "
	          << p->v_shape << std::endl;


}

} // namespace quakins

#endif /* _INITIALIZATION_HPP_ */
