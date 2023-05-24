/* -------------------------------
 *     Main of the Quakins Code
 * ------------------------------- */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <cstdio>
#include <mpi.h>
#include <nccl.h>
#include "include/initialization.hpp"
#include "include/PhaseSpaceInitialization.hpp"
#include "include/Timer.hpp"
#include "include/ReorderCopy.hpp"

using Real = float;

constexpr std::size_t dim = 4;


int main(int argc, char* argv[]) {

	int mpi_size, mpi_rank, root_rank=0;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);

	Timer watch(mpi_rank);

	Parameters<std::size_t,Real,dim> *p = 
             new Parameters<std::size_t,Real,dim>;

	quakins::init(p,mpi_rank);

	ncclUniqueId id; // tell every rank this id
	ncclComm_t comm;
//	Real *electron_dev, *electron_dev_buff;
	thrust::device_vector<Real> electron_dev(p->n_1d_per_dev), 
	                            electron_dev_buff(p->n_1d_per_dev);
	cudaStream_t stream;

	if(mpi_rank==0) ncclGetUniqueId(&id);
	
	MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);


	cudaSetDevice(mpi_rank);
//	cudaMalloc(&electron_dev,sizeof(Real)*p->n_1d_per_dev);
//	cudaMalloc(&electron_dev_buff,sizeof(Real)*p->n_1d_per_dev);
	cudaStreamCreate(&stream);

	ncclCommInitRank(&comm, mpi_size, id, mpi_rank);

	quakins::PhaseSpaceInitialization
	        <std::size_t,Real,dim>  phaseSpaceInit(p);

	watch.tick("Phase space initialization directly on devices...");
	phaseSpaceInit(thrust::device, 
	               electron_dev.begin(),
	               p->n_1d_per_dev,p->n_1d_per_dev*mpi_rank);
	watch.tock();
/*	

	watch.tick("Copy from GPU to GPU...");
	if (mpi_rank==0)
		ncclSend(electron_dev,p->n_1d_per_dev,ncclFloat,1,comm,stream);
	if (mpi_rank==1)
		ncclRecv(electron_dev_buff,p->n_1d_per_dev,ncclFloat,0,comm,stream);

	cudaStreamSynchronize(stream);
	watch.tock();
*/

	watch.tick("Copy from GPU to CPU...");

	thrust::host_vector<Real> _f_electron(p->n_1d_per_dev);
	thrust::copy(electron_dev.begin(),electron_dev.end(),_f_electron.begin());

	std::ofstream fout("wf@"+std::to_string(mpi_rank)+".qout",std::ios::out);
	fout << _f_electron << std::endl;
	fout.close();

	watch.tock();
	
	ncclCommDestroy(comm);
	MPI_Finalize();

}

