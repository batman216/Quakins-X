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

int main(int argc, char* argv[]) {

	Timer watch;

	Parameters<std::size_t,Real,dim> *p = 
             new Parameters<std::size_t,Real,dim>;

	quakins::init(p);

	watch.tick("Creating phase space on the host...");
	thrust::host_vector<Real> _f_electron(p->n_1d_tot);
	watch.tock();

	quakins::PhaseSpaceInitialization phaseSpaceInit(p);
	phaseSpaceInit(_f_electron.begin(),_f_electron.end());


	std::vector<thrust::device_vector<Real>*> electron_p_devs;

	watch.tick("Copy to GPU devices...");
	for (int i=0; i<p->n_dev; i++) {
		cudaSetDevice(i);
		electron_p_devs.push_back(
		                new thrust::device_vector
		                <Real>{static_cast<std::size_t>
		                (p->n_1d_per_dev)});
		
		thrust::copy(_f_electron.begin()+i*p->n_1d_per_dev,
		             _f_electron.begin()+(i+1)*p->n_1d_per_dev, 
		             (*electron_p_devs[i]).begin());
		cudaDeviceSynchronize();
	}
	watch.tock();

}

