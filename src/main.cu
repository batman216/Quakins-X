/* -------------------------------
 *     Main of the Quakins Code
 * ------------------------------- */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <cstdio>
#include <nccl.h>
#include "include/initialization.hpp"
#include "include/PhaseSpaceInitialization.hpp"
#include "include/Timer.hpp"

using Real = float;

constexpr std::size_t dim = 4;

/*
template <typename idx_type, typename val_type, idx_type dim>
struct dTestShape {

	typedef std::array<idx_type,dim> idx_array;
	typedef std::array<val_type,dim> val_array;

	idx_array n_dim;
	val_array low_bound, h;

	template <typename ExecutionPolicy>
	dTestShape(const ExecutionPolicy& exec,
	           idx_array n_dim, val_array lb, val_array h)
	 : n_dim(n_dim),low_bound(lb),h(h), exec(exec) {}

	__host__ __device__ 
	val_type operator()(const idx_type& idx) {

		idx_type imod, idvd;
		idx_array idx_m;
		for (int i=0; i<dim; i++) {
			
			imod = thrust::reduce(exec,
			                      n_dim.begin(),n_dim.begin()+i+1,1,
			                      thrust::multiplies<idx_type>());
			idvd = thrust::reduce(exec,
			                      n_dim.begin(),n_dim.begin()+i,1,
			                      thrust::multiplies<idx_type>());
			idx_m[i] = (idx % imod) / idvd;

		}

		return   std::exp(-std::pow(low_bound[0]+h[0]*idx_m[0]-5,2))
		       * std::exp(-std::pow(low_bound[1]+h[1]*idx_m[1]-5,2))
		       * std::exp(-std::pow(low_bound[2]+h[2]*idx_m[2],2))
		       * std::exp(-std::pow(low_bound[3]+h[3]*idx_m[3],2));

	}

};
*/

int main(int argc, char* argv[]) {

	Timer watch;

	Parameters<std::size_t,Real,dim> *p = 
             new Parameters<std::size_t,Real,dim>;

	quakins::init(p);

	std::vector<thrust::device_vector<Real>*> electron_p_devs;
	for (int i=0; i<p->n_dev; i++) {
		cudaSetDevice(i);
		electron_p_devs.push_back(new thrust::device_vector
		                <Real>{static_cast<std::size_t>(p->n_1d_per_dev)});
	}
	watch.tick("Allocating memory for phase space on the host...");
	thrust::host_vector<Real> _f_electron(p->n_1d_tot);
	watch.tock();



	quakins::PhaseSpaceInitialization
	        <std::size_t,Real,dim>  phaseSpaceInit(p);
	
	watch.tick("Phase space initialization directly on devices...");
	#pragma omp parallel for
	for (int i=0; i<p->n_dev; i++) {
		cudaSetDevice(i);
		phaseSpaceInit(thrust::device, electron_p_devs[i]->begin(),
		               p->n_1d_per_dev,p->n_1d_per_dev*i);
	}
	watch.tock();

	#pragma omp parallel for
	for (int i=0; i<p->n_dev; i++) {
		cudaSetDevice(i);
		thrust::copy(electron_p_devs[i]->begin(),electron_p_devs[i]->end(),
		             _f_electron.begin() + i*(p->n_1d_per_dev));

	}
	std::ofstream fout("wf",std::ios::out);
	std::copy(_f_electron.begin(), _f_electron.end(),
	          std::ostream_iterator<Real>(fout,"	"));
	fout.close();

}

