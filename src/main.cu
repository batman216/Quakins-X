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
#include "include/BoundaryCondition.hpp"
#include "include/Integrator.hpp"
#include "include/PoissonSolver.hpp"

using Nums = std::size_t;
using Real = float;


constexpr Nums dim = 4;


int main(int argc, char* argv[]) {

  Timer watch;

  Parameters<Nums,Real,dim> *p = 
             new Parameters<Nums,Real,dim>;

  try { quakins::init(p); }
  catch (std::invalid_argument& e) 
  {
    std::cerr << e.what() << std::endl;
#ifndef ANYHOW
    return -1;
#endif
  }
  int devs[p->n_dev];
  std::iota(devs,devs+p->n_dev,0);

  
  Nums nx1=p->n[2], nx2=p->n[3];
  Nums nx1bd=p->n_ghost[2], nx2bd=p->n_ghost[3];
  Nums nx1tot=p->n_tot[2], nx2tot=p->n_tot_local[3];
  Nums nv1=p->n[0], nv2=p->n[1];
  Real v1min=p->low_bound[0], v1max=p->up_bound[0];
  Real v2min=p->low_bound[1], v2max=p->up_bound[1];
  Real x1min=p->low_bound[2], x1max=p->up_bound[2];
  Real x2min=p->low_bound[3], x2max=p->up_bound[3];

  std::vector<thrust::device_vector<Real>*> 
    l_send_buff, l_recv_buff, r_send_buff, r_recv_buff, 
    f_e, f_e_buff,intg_buff, dens_e;

  Nums comm_size = p->n_ghost[3]*nx1tot*nv1*nv2;
  for (auto id : devs) {
    cudaSetDevice(id);
    f_e.push_back(new thrust::device_vector
                    <Real>{p->n_1d_per_dev});
    f_e_buff.push_back(new thrust::device_vector
                    <Real>{p->n_1d_per_dev});
    intg_buff.push_back(new thrust::device_vector
                    <Real>{p->n_1d_per_dev/nv1});
    dens_e.push_back(new thrust::device_vector
                    <Real>{p->n_1d_per_dev/nv1/nv2});
    l_send_buff.push_back(new thrust::device_vector
                    <Real>{comm_size});
    l_recv_buff.push_back(new thrust::device_vector
                    <Real>{comm_size});
    r_send_buff.push_back(new thrust::device_vector
                    <Real>{comm_size});
    r_recv_buff.push_back(new thrust::device_vector
                    <Real>{comm_size});
  }

  quakins::PhaseSpaceInitialization
          <Nums,Real,dim>  phaseSpaceInit(p);

  std::array<Nums,4> order1 = {2,3,1,0},
                     order2 = {1,0,3,2},
                     order3 = {2,3,1,0};

  std::array<Nums,4> n_now_1 = p->n_tot_local;
  std::array<Nums,4> n_now_2, n_now_3, n_now_4;
  quakins::ReorderCopy<Nums,Real,dim> copy1(n_now_1,order1);
  thrust::gather(order1.begin(),order1.end(),
                 n_now_1.begin(), n_now_2.begin());
  quakins::ReorderCopy<Nums,Real,dim> copy2(n_now_2,order2);
  thrust::gather(order2.begin(),order2.end(),
                 n_now_2.begin(), n_now_3.begin());
  quakins::ReorderCopy<Nums,Real,dim> copy3(n_now_3,order3);
  thrust::gather(order3.begin(),order3.end(),
                 n_now_3.begin(), n_now_4.begin());

  std::copy(n_now_2.begin(),n_now_2.end(),
            std::ostream_iterator<Nums>(std::cout," "));
  std::cout << std::endl;
  std::copy(n_now_3.begin(),n_now_3.end(),
            std::ostream_iterator<Nums>(std::cout," "));
  std::cout << std::endl;
  std::copy(n_now_4.begin(),n_now_4.end(),
            std::ostream_iterator<Nums>(std::cout," "));
  std::cout << std::endl;


  FreeStream<Nums,Real,dim,2,0> fsSolverX1(p,p->dt*.5);
  FreeStream<Nums,Real,dim,3,1> fsSolverX2(p,p->dt*.5);

  quakins::BoundaryCondition<Nums,PeriodicBoundary>
    boundX1(nx1,nx1bd);

  quakins::BoundaryCondition<Nums,PeriodicBoundaryPara>
    boundX2(nx2,nx2bd);

  quakins::PoissonSolver<Nums,Real,2> 
    poissonSolver({nx1,nx2},{x1min,x2min, x1max,x2max});

  quakins::Integrator<Real> 
    integral1(nv1,p->n_1d_per_dev/nv1,v1min,v1max);
  quakins::Integrator<Real> 
    integral2(nv2,p->n_1d_per_dev/nv1/nv2,v2min,v2max);
  thrust::host_vector<Real> _dens_e(p->n_1d_tot/nv1/nv2);


  watch.tick("Phase space initialization directly on devices..."); //------
  #pragma omp parallel for
  for (auto id : devs) {
    cudaSetDevice(id);
    phaseSpaceInit(thrust::device, 
                   f_e[id]->begin(), p->n_1d_per_dev,id);
    integral1(f_e[id]->begin(),intg_buff[id]->begin());
    integral2(intg_buff[id]->begin(),dens_e[id]->begin());
    
    thrust::copy(dens_e[id]->begin(),dens_e[id]->end(),
               _dens_e.begin() + id*nx1tot*nx2tot);

  }

  std::ofstream d0out("init_dens.qout",std::ios::out);
  d0out << _dens_e;
  d0out.close();

  watch.tock(); //=========================================================
  ncclComm_t comm[p->n_dev];
  cudaStream_t *stream = (cudaStream_t *) malloc(sizeof(cudaStream_t) * 2);
  ncclCommInitAll(comm,p->n_dev,devs);
  for (auto id : devs) {
    cudaSetDevice(id);
    cudaStreamCreate(stream+id);
  }    
  watch.tick("Main Loop start..."); //------------------------------------
  std::string flag = {'l','r'}; 
  int rank_end = p->n_dev-1;
  std::ofstream dout("dens_e.qout",std::ios::out);
  for (Nums step=0; step<p->time_step_total; step++) {
    for (auto id : devs) {
      cudaSetDevice(id);
      thrust::copy(f_e[id]->end()-2*comm_size,f_e[id]->end()-comm_size, 
                   r_send_buff[id]->begin());
      thrust::copy(f_e[id]->begin()+comm_size,f_e[id]->begin()+2*comm_size, 
                   l_send_buff[id]->begin());
      cudaStreamCreate(stream+id);
      cudaStreamSynchronize(stream[id]);
    }
    watch.tick("NCCL communicating..."); //----------------------------------
    ncclGroupStart();// <--
    for (int id = 1; id<=rank_end; id++) {
      ncclSend(thrust::raw_pointer_cast(l_send_buff[id]->data()),
             comm_size, ncclFloat, id-1, comm[id], stream[id]); 
      ncclRecv(thrust::raw_pointer_cast(r_recv_buff[id-1]->data()),
             comm_size, ncclFloat, id, comm[id-1], stream[id-1]); 
    }
    ncclSend(thrust::raw_pointer_cast(l_send_buff[0]->data()),
             comm_size, ncclFloat, rank_end, comm[0], stream[0]); 
    ncclRecv(thrust::raw_pointer_cast(r_recv_buff[rank_end]->data()),
             comm_size, ncclFloat, 0, comm[rank_end], stream[rank_end]); 
    ncclGroupEnd();

    ncclGroupStart();// -->
    for (int id = 0; id<rank_end; id++) {
      ncclSend(thrust::raw_pointer_cast(r_send_buff[id]->data()),
               comm_size, ncclFloat, id+1, comm[id], stream[id]); 
      ncclRecv(thrust::raw_pointer_cast(l_recv_buff[id+1]->data()),
               comm_size, ncclFloat, id, comm[id+1], stream[id+1]); 
    }
    ncclSend(thrust::raw_pointer_cast(r_send_buff[rank_end]->data()),
             comm_size, ncclFloat, 0, comm[rank_end], stream[rank_end]); 
    ncclRecv(thrust::raw_pointer_cast(l_recv_buff[0]->data()),
             comm_size, ncclFloat, rank_end, comm[0], stream[0]); 
    ncclGroupEnd();
    for (auto id : devs) {
      cudaSetDevice(id);
      cudaStreamSynchronize(stream[id]);
    }  
    watch.tock(); //=========================================================

    watch.tick("pushing..."); //----------------------------------
    #pragma omp parallel for
    for (auto id : devs) {
      cudaSetDevice(id);
      cudaStreamSynchronize(stream[id]);
      thrust::copy(l_recv_buff[id]->begin(),l_recv_buff[id]->end(),
                   f_e[id]->begin());
      thrust::copy(r_recv_buff[id]->begin(),r_recv_buff[id]->end(),
                   f_e[id]->end()-comm_size);

      copy1(f_e[id]->begin(),f_e[id]->end(),f_e_buff[id]->begin());
      boundX1(f_e_buff[id]->begin(),f_e_buff[id]->end(),flag[id]);
      fsSolverX1(f_e_buff[id]->begin(),
                 f_e_buff[id]->end(),
                 p->n_1d_per_dev/p->n_tot_local[0],id);

      copy2(f_e_buff[id]->begin(),f_e_buff[id]->end(),f_e[id]->begin());
      //boundX2(f_e[id]->begin(),f_e[id]->end(),flag[id]);
      fsSolverX2(f_e[id]->begin(),
                 f_e[id]->end(),
                 p->n_1d_per_dev/p->n_tot_local[1],id);
      copy3(f_e[id]->begin(),f_e[id]->end(),f_e_buff[id]->begin());
      thrust::copy(f_e_buff[id]->begin(),f_e_buff[id]->end(),f_e[id]->begin());
      cudaStreamSynchronize(stream[id]);

      integral1(f_e[id]->begin(),intg_buff[id]->begin());
      integral2(intg_buff[id]->begin(),dens_e[id]->begin());
    
      thrust::copy(dens_e[id]->begin(),dens_e[id]->end(),
                   _dens_e.begin() + id*nx1tot*nx2tot);
    }
    watch.tock(); //==========================================================

   

    if (step%(p->dens_print_intv)==0)
      dout << _dens_e << std::endl;
  }
  watch.tock(); //============================================================

  
  dout.close();

/*
  watch.tick("Copy from GPU to CPU...");
  #pragma omp parallel for
  for (int i=0; i<p->n_dev; i++) {
    cudaSetDevice(i);
    thrust::copy(f_e_buff[i]->begin(),f_e_buff[i]->end(),
                 _f_electron.begin() + i*(p->n_1d_per_dev));

  }
  watch.tock();


  watch.tick("Allocating memory for phase space on the host...");
  thrust::host_vector<Real> _f_electron;
  _f_electron.resize(p->n_1d_tot);
  watch.tock();
  std::ofstream fout("wf.qout",std::ios::out);
  fout << _f_electron ;
  fout.close();
  */
}


