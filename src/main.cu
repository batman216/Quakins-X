/* -------------------------------
 *     Main of the Quakins Code
 * ------------------------------- */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <cstdio>
#include <mpi.h>
#include <nccl.h>
#include <omp.h>
#include "include/initialization.hpp"
#include "include/PhaseSpaceInitialization.hpp"

using Real = float;

constexpr int dim = 4;

int main(int argc, char* argv[]) {

	std::ifstream input_file("quakins.input");
	auto domain_map = read_box(input_file, "domain");


	Parameters<int,Real,dim> *p = new Parameters<int,Real,dim>;
	
	quakins::initDomain(p,domain_map);
	thrust::host_vector<Real> _f_electron(p->n_1d_tot);

	int n_dev;
	cudaGetDeviceCount(&n_dev);
	std::cout << "The Wigner function costs " 
	          << p->n_1d_tot*sizeof(Real)/1048576 << "Mb of Memory" << std::endl;

	std::cout << "You have " << omp_get_num_procs() << " CPU hosts." << std::endl;
	std::cout << n_dev << " GPU devices are found: " << std::endl;
	for (int i=0; i<n_dev; i++) {
		cudaDeviceProp dev_prop;
		cudaGetDeviceProperties(&dev_prop,i);
		std::cout << "	" << i << ": " << dev_prop.name << std::endl;
	}

	std::vector<thrust::device_vector<Real>*> electron_p_devs;
	for (int i=0; i<n_dev; i++) {
		cudaSetDevice(i);
		electron_p_devs.push_back(new thrust::device_vector<Real>{p->n_1d_tot});	
	}
	//ncclGroupStart();
//	omp_set_num_threads(n_dev);
//	#pragma omp parallel
	for (int i=0; i<n_dev; i++)
	{
		cudaSetDevice(i);
		*electron_p_devs[i] = _f_electron;
		cudaDeviceSynchronize();
	}
	//ncclGroupEnd();
	

	quakins::PhaseSpaceInitialization phaseSpaceInit(p);

	phaseSpaceInit(_f_electron.begin(),_f_electron.end());


}

