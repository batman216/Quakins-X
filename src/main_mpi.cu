/* -------------------------------
 *     Main of the Quakins-X Code
 * ------------------------------- */

#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <cstdio>
#include <mpi.h>
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
#include "include/ParallelCommunicator.hpp"
#include "include/InsertGhost.hpp"


using Nums = std::size_t;
using Real = float;


constexpr Nums dim = 4;

#define MCW MPI_COMM_WORLD

int main(int argc, char* argv[]) {

  int mpi_rank, mpi_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MCW, &mpi_rank);
  MPI_Comm_size(MCW, &mpi_size);

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
  Nums nx1tot=p->n_all[2], nx2allloc=p->n_all_local[3];
  Nums nv1=p->n[0], nv2=p->n[1];
  Nums nxall = nx1tot*nx2allloc;
  Nums comm_size = p->n_ghost[3]*nx1tot*nv1*nv2;
  Nums dens_size = nx1tot*nx2/p->n_dev;
  
  Real v1min=p->low_bound[0], v1max=p->up_bound[0];
  Real v2min=p->low_bound[1], v2max=p->up_bound[1];
  Real x1min=p->low_bound[2], x1max=p->up_bound[2];
  Real x2min=p->low_bound[3], x2max=p->up_bound[3];

  
  quakins::ParallelCommunicator<Nums,Real> 
    q_comm(mpi_rank, mpi_size, MCW);
  quakins::PhaseSpaceParallelCommute<Nums,Real>
    psCommute(comm_size, &q_comm);
  quakins::DensityGather<Nums,Real>
    densGather(nx1tot*nx2loc, &q_comm);
  quakins::PotentialBroadcast<Nums,Real>
    potBcast(nx1*nx2,&q_comm);

  thrust::device_vector<Real> 
    f_e(p->n_1d_per_dev), f_e_buff(p->n_1d_per_dev);
  thrust::device_vector<Real> 
    intg_buff(nxall*nv2), dens_e(nxall);
  std::array<thrust::device_vector<Real>,dim/2> E;
  for (int i=0; i<dim/2; i++) E[i].resize(nx1*nx2);


  thrust::device_vector<Real> 
    dens_e_all(nx1tot*nx2), dens_e_all_buff(nx1tot*nx2), pote_all(nx1*nx2),
    pote_all_tot(nx1tot*nx2);
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

  quakins::ReorderCopy<Nums,Real,dim/2> dens_copy({nx1tot,nx2},{1,0});

  quakins::FreeStream<Nums,Real,dim,2,0,
    quakins::details::FluxBalanceCoordSpace> fsSolverX1(p,p->dt*.5);
  quakins::FreeStream<Nums,Real,dim,3,1,
    quakins::details::FluxBalanceCoordSpace> fsSolverX2(p,p->dt*.5);
  quakins::FreeStream<Nums,Real,dim,2,0,
    quakins::details::WignerTerm> wignerSolver(p,p->dt);
  quakins::FreeStream<Nums,Real,dim,4,0,
    quakins::details::FourierSpectrumVeloSpace> vSolver(p,p->dt);
 

  quakins::BoundaryCondition<Nums,ReflectingBoundary>
    boundX1(nx1,nx1bd,nv1,nx1tot*nx2allloc*nv2);


  quakins::PoissonSolver<Nums,Real,2, FFTandInvHost> 
    poissonSolver({nx1,nx2},{x1min,x2min, x1max,x2max});

  quakins::Integrator<Real> 
    integral1(nv1,p->n_1d_per_dev/nv1,v1min,v1max);
  quakins::Integrator<Real> 
    integral2(nv2,p->n_1d_per_dev/nv1/nv2,v2min,v2max);

//--------------------------------------------------------------------
  quakins::PhaseSpaceInitialization
          <Nums,Real,dim,ShapeFunctor>  phaseSpaceInit(p);
  phaseSpaceInit(thrust::device, 
                 f_e.begin(), p->n_1d_per_dev, mpi_rank);
  integral1(f_e.begin(),intg_buff.begin());
  integral2(intg_buff.begin(),dens_e.begin());
    

  std::ofstream dout("dens_e@"+std::to_string(mpi_rank)+".qout",std::ios::out);
  std::ofstream pout("potential@"+std::to_string(mpi_rank)+".qout",std::ios::out);
  

  densGather(dens_e_all.begin(), dens_e.begin()+nx1tot*nx2bd);


  dens_copy(dens_e_all.begin(),dens_e_all.end(),dens_e_all_buff.begin());
  thrust::copy(dens_e_all_buff.begin()+nx2*nx1bd,
               dens_e_all_buff.end()-nx2*nx1bd,
               _dens_e_all.begin());
  poissonSolver(_dens_e_all.begin(),_dens_e_all.end(),_pote_all.begin());
 


  Nums id = mpi_rank;
  
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

    nccl_watch.tick("NCCL communicating..."); //----------------------------------
    psCommute(f_e.begin(),f_e.end());
    nccl_watch.tock(); //=========================================================
      
    push_watch.tick("--> step[" +std::to_string(step)+ "] pushing..."); //--------
    copy1(f_e.begin(), f_e.end(),f_e_buff.begin()); // n_now = {nx2l,nx1,nv1,nv2}
    fsSolverX2(f_e_buff.begin(), f_e_buff.end(),
               thrust::make_discard_iterator(), id);
    copy2(f_e_buff.begin(),f_e_buff.end(),f_e.begin()); // n_now = {nx1,nx2l,nv2,nv1}

    boundX1(f_e.begin(),f_e.end(),flag);
    fsSolverX1(f_e.begin(), f_e.end(), 
               thrust::make_discard_iterator(), id);
    copy3(f_e.begin(),f_e.end(),f_e_buff.begin()); // n_now = {nv1,nv2,nx1,nx2l}

    thrust::copy(f_e_buff.begin(),f_e_buff.end(),f_e.begin());

    push_watch.tock(); //========================================================

    poi_watch.tick("solving poisson..."); //-------------------------------------

    integral1(f_e.begin(),intg_buff.begin());
    integral2(intg_buff.begin(),dens_e.begin());
    
    densGather(dens_e_all.begin(), dens_e.begin()+nx1tot*nx2bd);

    if (mpi_rank==0) {
      dens_copy(dens_e_all.begin(),dens_e_all.end(),dens_e_all_buff.begin());
      thrust::copy(dens_e_all_buff.begin()+nx2*nx1bd,
                   dens_e_all_buff.end()-nx2*nx1bd, _dens_e_all.begin());

      poissonSolver(_dens_e_all.begin(),_dens_e_all.end(),_pote_all.begin());

      thrust::copy(_pote_all.begin(),_pote_all.end(),pote_all.begin());    
    }
    //potBcast(pote_all.begin());

    //  quakins::insertGhost(pote_all.begin(),nx1,nx1bd,pote_all_tot.begin(),nx2);
    
    poi_watch.tock();  //========================================================

    if (step%(p->dens_print_intv)==0) {
      dout << _dens_e_all << std::endl;
      pout << _pote_all << std::endl;
    }
    // velocity direction push  
    
    vSolver(f_e.begin(), f_e.end(),pote_all.begin(),id);

  }

  the_watch.tock();
  dout.close();

}


