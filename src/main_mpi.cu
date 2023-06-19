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
#include "include/util.hpp"
#include "include/details/free_stream_algorithm.hpp"


using Nums = std::size_t;
using Real = float;


constexpr Nums dim = 4;

#define MCW MPI_COMM_WORLD

int main(int argc, char* argv[]) {

  int mpi_rank, mpi_size, local_rank=0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MCW, &mpi_rank);
  MPI_Comm_size(MCW, &mpi_size);

  uint64_t host_hashs[mpi_size];
  char hostname[1024];
  getHostName(hostname, 1024);
  host_hashs[mpi_rank] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,host_hashs,
                sizeof(uint64_t), MPI_BYTE, MCW);
  for (int p=0; p<mpi_size; p++) {
    if (p==mpi_rank) break;
    if (host_hashs[p]==host_hashs[mpi_rank]) local_rank++;
  }

  ncclUniqueId nccl_id;
  ncclComm_t comm;
  cudaStream_t s;

  if (mpi_rank==0) ncclGetUniqueId(&nccl_id);
  MPI_Bcast((void*)&nccl_id,sizeof(nccl_id),MPI_BYTE,0,MCW);
  
  cudaSetDevice(local_rank);
  cudaStreamCreate(&s);
  
  ncclCommInitRank(&comm,mpi_size,nccl_id,mpi_rank);

  Parameters<Nums,Real,dim> *p = 
             new Parameters<Nums,Real,dim>;

  try { quakins::init(p, mpi_rank); }
  catch (std::invalid_argument& e) 
  {
    std::cerr << e.what() << std::endl;
#ifndef ANYHOW
    return -1;
#endif
  }
  
  Nums nx1=p->n[2], nx2=p->n[3], nx2loc=nx2/p->n_dev;
  Nums nx1bd=p->n_ghost[2], nx2bd=p->n_ghost[3];
  Nums nx1all=p->n_all[2], nx2allloc=p->n_all_local[3];
  Nums nv1=p->n[0], nv2=p->n[1];
  Nums nxall = nx1all*nx2allloc;
  Nums comm_size = p->n_ghost[3]*nx1all*nv1*nv2;
  Nums dens_size = nx1all*nx2/p->n_dev;
  
  Real v1min=p->low_bound[0], v1max=p->up_bound[0];
  Real v2min=p->low_bound[1], v2max=p->up_bound[1];
  Real x1min=p->low_bound[2], x1max=p->up_bound[2];
  Real x2min=p->low_bound[3], x2max=p->up_bound[3];

  thrust::device_vector<Real> 
    l_send_buff(comm_size), l_recv_buff(comm_size), 
    r_send_buff(comm_size), r_recv_buff(comm_size); 
  thrust::device_vector<Real> 
    f_e(p->n_1d_per_dev), f_e_buff(p->n_1d_per_dev);
  thrust::device_vector<Real> 
    intg_buff(nxall*nv2), dens_e(nxall);
  thrust::device_vector<Real> 
    dens_e_all(nx1all*nx2), dens_e_all_buff(nx1all*nx2), pote_all(nx1*nx2);
  thrust::host_vector<Real> _dens_e_all(nx1*nx2), _pote_all(nx1*nx2);

  std::array<Nums,4> order1 = {3,2,0,1},
                     order2 = {1,0,3,2},
                     order3 = {3,2,0,1};

  std::array<Nums,4> n_now_1 = p->n_all_local;
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

  quakins::ReorderCopy<Nums,Real,dim/2> dens_copy({nx1all,nx2},{1,0});

  quakins::FreeStream<Nums,Real,dim,2,0,
    quakins::details::FluxBalance> fsSolverX1(p,p->dt*.5);
  quakins::FreeStream<Nums,Real,dim,3,1,
    quakins::details::FluxBalance> fsSolverX2(p,p->dt*.5);

  quakins::BoundaryCondition<Nums,ReflectingBoundary>
    boundX1(nx1,nx1bd,nv1,nx1all*nx2allloc*nv2);


  quakins::PoissonSolver<Nums,Real,2, FFTandInvHost> 
    poissonSolver({nx1,nx2},{x1min,x2min, x1max,x2max});

  quakins::Integrator<Real> 
    integral1(nv1,p->n_1d_per_dev/nv1,v1min,v1max);
  quakins::Integrator<Real> 
    integral2(nv2,p->n_1d_per_dev/nv1/nv2,v2min,v2max);

//--------------------------------------------------------------------
  quakins::PhaseSpaceInitialization
          <Nums,Real,dim>  phaseSpaceInit(p);
  phaseSpaceInit(thrust::device, 
                 f_e.begin(), p->n_1d_per_dev, mpi_rank);
  integral1(f_e.begin(),intg_buff.begin());
  integral2(intg_buff.begin(),dens_e.begin());
    

  std::ofstream dout("dens_e@"+std::to_string(mpi_rank)+".qout",std::ios::out);
  std::ofstream pout("potential@"+std::to_string(mpi_rank)+".qout",std::ios::out);
  //dout << dens_e << std::endl;

  ncclGroupStart();
  if (mpi_rank==0) {
    for (int r=0; r<mpi_size; r++)
      ncclRecv(thrust::raw_pointer_cast(dens_e_all.data())
               +r*dens_size, dens_size,ncclFloat,r,comm,s);
  }
  ncclSend(thrust::raw_pointer_cast(dens_e.data())
           +nx1all*nx2bd, dens_size,ncclFloat,0,comm,s);
  ncclGroupEnd();
 
  if (mpi_rank==0) {
    dens_copy(dens_e_all.begin(),dens_e_all.end(),dens_e_all_buff.begin());
    thrust::copy(dens_e_all_buff.begin()+nx2*nx1bd,
                 dens_e_all_buff.end()-nx2*nx1bd,
                 _dens_e_all.begin());
    poissonSolver(_dens_e_all.begin(),_dens_e_all.end(),_pote_all.begin());
    dout << _dens_e_all << std::endl;
    pout << _pote_all << std::endl;
 
  }


  Nums id = mpi_rank;
  Nums l_rank = id==0? mpi_size-1 : id-1;
  Nums r_rank = id==mpi_size-1? 0 : id+1;
  
  char flag;
  if (id==0) flag='l'; 
  else if (id==mpi_size-1) flag='r';
  else flag='m';
  
  Timer the_watch(mpi_rank,"This run");
  Timer push_watch(mpi_rank,"push");
  Timer nccl_watch(mpi_rank,"nccl communination");
  Timer poi_watch(mpi_rank,"solver Poisson equation");
  /*
  std::ofstream fout("ftest@" +std::to_string(mpi_rank)+ ".qout",std::ios::out);
  fout << f_e << std::endl;
*/
  the_watch.tick("Main Loop start...");
  for (Nums step=0; step<p->time_step_total; step++) {

    thrust::copy(f_e.end()-2*comm_size,f_e.end()-comm_size, 
                 r_send_buff.begin());
    thrust::copy(f_e.begin()+comm_size,f_e.begin()+2*comm_size, 
                 l_send_buff.begin());
    nccl_watch.tick("NCCL communicating..."); //----------------------------------
    ncclGroupStart();// <--
    ncclSend(thrust::raw_pointer_cast(l_send_buff.data()),
             comm_size, ncclFloat, l_rank, comm, s); 
    ncclRecv(thrust::raw_pointer_cast(r_recv_buff.data()),
             comm_size, ncclFloat, r_rank, comm, s); 
    ncclGroupEnd();

    ncclGroupStart();// -->
    ncclSend(thrust::raw_pointer_cast(r_send_buff.data()),
             comm_size, ncclFloat, r_rank, comm, s); 
    ncclRecv(thrust::raw_pointer_cast(l_recv_buff.data()),
             comm_size, ncclFloat, l_rank, comm, s); 
    ncclGroupEnd();

    cudaStreamSynchronize(s);

    
    thrust::copy(l_recv_buff.begin(),l_recv_buff.end(),
                 f_e.begin());
    thrust::copy(r_recv_buff.begin(),r_recv_buff.end(),
                 f_e.end()-comm_size);
    nccl_watch.tock(); //=========================================================
      
    push_watch.tick("--> step[" +std::to_string(step)+ "] pushing..."); //--------
    copy1(f_e.begin(), f_e.end(),f_e_buff.begin()); // n_now = {nx2l,nx1,nv1,nv2}
    fsSolverX2(f_e_buff.begin(),
               f_e_buff.end(),
               p->n_1d_per_dev/p->n_all_local[1],id);
    copy2(f_e_buff.begin(),f_e_buff.end(),f_e.begin()); // n_now = {nx1,nx2l,nv2,nv1}

    boundX1(f_e.begin(),f_e.end(),flag);
    fsSolverX1(f_e.begin(),
               f_e.end(),
               p->n_1d_per_dev/p->n_all_local[0],id);
    copy3(f_e.begin(),f_e.end(),f_e_buff.begin()); // n_now = {nv1,nv2,nx1,nx2l}

    thrust::copy(f_e_buff.begin(),f_e_buff.end(),f_e.begin());

    cudaStreamSynchronize(s);

    integral1(f_e.begin(),intg_buff.begin());
    integral2(intg_buff.begin(),dens_e.begin());
    
    push_watch.tock(); //==========================================================

    poi_watch.tick("solving poisson...");

    ncclGroupStart();
    if (mpi_rank==0) {
      for (int r=0; r<mpi_size; r++)
        ncclRecv(thrust::raw_pointer_cast(dens_e_all.data())
                 +r*dens_size, dens_size,ncclFloat,r,comm,s);
    }
    ncclSend(thrust::raw_pointer_cast(dens_e.data())
             +nx1all*nx2bd, dens_size,ncclFloat,0,comm,s);
    ncclGroupEnd();
  
    if (mpi_rank==0) {
      dens_copy(dens_e_all.begin(),dens_e_all.end(),dens_e_all_buff.begin());
      thrust::copy(dens_e_all_buff.begin()+nx2*nx1bd,
                   dens_e_all_buff.end()-nx2*nx1bd,
                   _dens_e_all.begin());
      poissonSolver(_dens_e_all.begin(),_dens_e_all.end(),_pote_all.begin());
    }
    poi_watch.tock();

    if (mpi_rank==0 && step%(p->dens_print_intv)==0) {
      dout << _dens_e_all << std::endl;
      pout << _pote_all << std::endl;
    }
  }
  

  the_watch.tock();
  dout.close();


}


