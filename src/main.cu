/* -------------------------------
 *     Main of the Quakins-X Code
 * ------------------------------- */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <cstdio>
#include <mpi.h>
#include <nccl.h>
#include <cufftXt.h>

#include "include/initialization.hpp"
#include "include/PhaseSpaceInitialization.hpp"
#include "include/Timer.hpp"
#include "include/ReorderCopy.hpp"
#include "include/FreeStream.hpp"
#include "include/Integrator.hpp"

using Real = float;

constexpr std::size_t dim = 4;


int main(int argc, char* argv[]) {

  Timer watch;

  Parameters<std::size_t,Real,dim> *p = 
             new Parameters<std::size_t,Real,dim>;

  try { quakins::init(p); }
  catch (std::invalid_argument& e) 
  {
    std::cerr << e.what() << std::endl;
#ifndef ANYHOW
    return -1;
#endif
  }
  
  std::size_t nx1=p->n[2], nx2=p->n[3];
  std::size_t nv1=p->n[0], nv2=p->n[1];
  Real v1min=p->low_bound[0], v1max=p->up_bound[0];
  Real v2min=p->low_bound[1], v2max=p->up_bound[1];

  std::vector<thrust::device_vector<Real>*> f_e, f_e_buff,intg_buff, dens_e;

  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);
    f_e.push_back(new thrust::device_vector
                    <Real>{static_cast<std::size_t>(p->n_1d_per_dev)});
    f_e_buff.push_back(new thrust::device_vector
                    <Real>{static_cast<std::size_t>(p->n_1d_per_dev)});
    intg_buff.push_back(new thrust::device_vector
                    <Real>{static_cast<std::size_t>(p->n_1d_per_dev/nv1)});
    dens_e.push_back(new thrust::device_vector
                    <Real>{static_cast<std::size_t>(p->n_1d_per_dev/nv1/nv2)});

  }

  quakins::PhaseSpaceInitialization
          <std::size_t,Real,dim>  phaseSpaceInit(p);

  std::array<std::size_t,4> order1 = {2,3,1,0},
                            order2 = {3,2,0,1};

  std::array<std::size_t,4> n_now = p->n_tot_local;
  quakins::ReorderCopy<std::size_t,Real,dim> copy1(n_now,order1);
  
  thrust::gather(order1.begin(),order1.end(),p->n_tot_local.begin(),
                n_now.begin());

  quakins::ReorderCopy<std::size_t,Real,dim> copy2(n_now,order2);
  
  FreeStream<std::size_t,Real,dim,2,0> fsSolverX1(p,p->dt*.5);
  FreeStream<std::size_t,Real,dim,3,1> fsSolverX2(p,p->dt*.5);

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

  quakins::Integrator<Real> 
    integral1(nv1,p->n_1d_per_dev/nv1,v1min,v1max);
  quakins::Integrator<Real> 
    integral2(nv2,p->n_1d_per_dev/nv1/nv2,v2min,v2max);
  thrust::host_vector<Real> _dens_e(p->n_1d_tot/nv1/nv2);

  watch.tick("Copy from GPU to CPU..."); //-------------------------------
  #pragma omp parallel for
  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);
    copy2(f_e_buff[i]->begin(),f_e_buff[i]->end(),f_e[i]->begin());

    thrust::copy(f_e[i]->begin(),f_e[i]->end(),
                 _f_electron.begin() + i*(p->n_1d_per_dev));

    integral1(f_e[i]->begin(),intg_buff[i]->begin());
    integral2(intg_buff[i]->begin(), dens_e[i]->begin());
        
    thrust::copy(dens_e[i]->begin(),dens_e[i]->end(),
                 _dens_e.begin() + i*(p->n_1d_per_dev/nv1/nv2));

  }
  watch.tock(); //========================================================

  watch.tick("Write to file..."); //--------------------------------------
  std::ofstream fbout("wfb.qout",std::ios::out);
  fbout << _f_electron ;
  fbout.close();
  watch.tock(); //========================================================
   
  
      watch.tick("Main Loop start..."); //-----------------------------------------------
  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);

    for (std::size_t step=0; step<p->time_step_total; step++) {
      fsSolverX1(f_e_buff[i]->begin(),
                 f_e_buff[i]->end(),
                 p->n_1d_per_dev/p->n_tot_local[0],i);

      copy2(f_e_buff[i]->begin(),f_e_buff[i]->end(),f_e[i]->begin());
    }
  }
  watch.tock(); //-------------------------------------------------------

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
    
    thrust::copy(f_e[i]->begin(),f_e[i]->end(),
                 _f_electron.begin() + i*(p->n_1d_per_dev));

  }
  watch.tock();

  std::ofstream dout("dens_e.qout",std::ios::out);
  dout << _dens_e ;
  dout.close();

  std::ofstream fout("wf.qout",std::ios::out);
  fout << _f_electron ;
  fout.close();

}

