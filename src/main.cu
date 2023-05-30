/* -------------------------------
 *     Main of the Quakins-X Code
 * ------------------------------- */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <mpi.h>
#include <nccl.h>
#include <cufftXt.h>

#include "include/initialization.hpp"
#include "include/PhaseSpaceInitialization.hpp"
#include "include/Timer.hpp"
#include "include/ReorderCopy.hpp"
#include "include/FreeStream.hpp"

using Real = float;

constexpr std::size_t dim = 4;


int main(int argc, char* argv[]) {

  Timer watch;

  Parameters<std::size_t,Real,dim> *p = 
             new Parameters<std::size_t,Real,dim>;

  quakins::init(p);

  std::vector<thrust::device_vector<Real>*> f_e;
  std::vector<thrust::device_vector<Real>*> f_e_buff;
  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);
    f_e.push_back(new thrust::device_vector
                    <Real>{static_cast<std::size_t>(p->n_1d_per_dev)});
    f_e_buff.push_back(new thrust::device_vector
                    <Real>{static_cast<std::size_t>(p->n_1d_per_dev)});
  }

  quakins::PhaseSpaceInitialization
          <std::size_t,Real,dim>  phaseSpaceInit(p);

  quakins::ReorderCopy<std::size_t,Real,dim> copy1(p->n_tot_local,{2,3,1,0});
  FreeStream<std::size_t,Real,dim,2,0> fsSolverX1(p,p->dt*.5);

  watch.tick("Phase space initialization directly on devices...");
  #pragma omp parallel for
  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);

    phaseSpaceInit(thrust::device, 
                   f_e[i]->begin(),
                   p->n_1d_per_dev,p->n_1d_per_dev*i);
    
    copy1(f_e[i]->begin(),f_e[i]->end(),f_e_buff[i]->begin());
  }
  watch.tock();

  watch.tick("Allocating memory for phase space on the host...");
  thrust::host_vector<Real> _f_electron;
  _f_electron.resize(p->n_1d_tot);
  watch.tock();

  watch.tick("Copy from GPU to CPU...");
  #pragma omp parallel for
  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);
    thrust::copy(f_e_buff[i]->begin(),f_e_buff[i]->end(),
                 _f_electron.begin() + i*(p->n_1d_per_dev));

  }
  watch.tock();

  std::ofstream fbout("wfb.qout",std::ios::out);
  fbout << _f_electron ;
  fbout.close();

   
  watch.tick("push...");
  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);
    for (std::size_t step=0; step<1000; step++) {
      fsSolverX1(f_e_buff[i]->begin(),
                 f_e_buff[i]->end(),
                 p->n_1d_per_dev/p->n_tot_local[0]);
    }
  }
  watch.tock();

  ncclComm_t comm[p->n_dev];
  cudaStream_t *stream = (cudaStream_t *) malloc(sizeof(cudaStream_t) * 2);
  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);
    cudaStreamCreate(stream+i);
  }
  int devs[2] = {0,1};
  ncclCommInitAll(comm,p->n_dev,devs);
  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);
    cudaStreamCreate(stream+i);
  }

/*
  watch.tick("Copy from GPU to GPU...");
  ncclGroupStart();
  ncclSend(thrust::raw_pointer_cast(f_e[1]->data()),
             p->n_1d_per_dev, ncclFloat, 0, comm[1], stream[1]); 
  ncclRecv(thrust::raw_pointer_cast(f_e_buff[0]->data()),
             p->n_1d_per_dev, ncclFloat, 1, comm[0], stream[0]); 

  ncclSend(thrust::raw_pointer_cast(f_e[0]->data()),
             p->n_1d_per_dev, ncclFloat, 1, comm[0], stream[0]); 
  ncclRecv(thrust::raw_pointer_cast(f_e_buff[1]->data()),
             p->n_1d_per_dev, ncclFloat, 0, comm[1], stream[1]); 
  ncclGroupEnd();
  for (int i = 0; i < p->n_dev; ++i) {
     cudaSetDevice(i);
     cudaStreamSynchronize(stream[i]);
  }
  watch.tock();
*/  


  watch.tick("Copy from GPU to CPU...");
  #pragma omp parallel for
  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);
    thrust::copy(f_e_buff[i]->begin(),f_e_buff[i]->end(),
                 _f_electron.begin() + i*(p->n_1d_per_dev));

  }
  watch.tock();

  std::ofstream fout("wf.qout",std::ios::out);
  fout << _f_electron ;
  fout.close();

}

