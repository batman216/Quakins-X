/**
 * @file      main_1d.cu
 * @brief     main file of the quakins 1d1v simualtion
 * @author    Tian-Xing Hu
 * @copyright 2023 Tian-Xing Hu
 */

#include <iostream>
#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>

//#include "include/initialization.hpp"
//#include "include/Timer.hpp"
//#include "include/util.hpp"
//#include "include/ParallelCommunicator.hpp"
//#include "include/InsertGhost.hpp"
//#include "include/fftOutPlace.hpp"
//#include "include/Slice.hpp"
//#include "include/TestParticle.hpp"
#include "include/Parameters.hpp"
#include "include/PhaseSpaceInitialization.hpp"
#include "include/Integrator.hpp"
#include "include/PoissonSolver.hpp"
#include "include/SplittingShift.hpp"
#include "include/ReorderCopy.hpp"
#include "include/Boundary.hpp"
#include "include/gizmos.hpp"
#include "include/diagnosis.hpp"
#include "include/QuantumSplittingShift.hpp"

using uInt = std::size_t;
using Real = float;


int main(int argc, char* argv[]) {

#ifdef PARA
  MPI_Init(&argc,&argv);
  int mpi_size,  mpi_rank;
  MPI_Comm_size(MCW,&mpi_size);
  MPI_Comm_rank(MCW,&mpi_rank);
  auto *p = new quakins::
    Parameters(mpi_rank,mpi_size);
#else
  auto *p = new quakins::
    Parameters<uInt,Real,DIM_X,DIM_V>();
#endif
  p->initial();
  
  uInt nx = p->n_main_x[0], nv = p->n_main_v[0], 
       nxbd = p->n_ghost_x[0],
       nvbd = p->n_ghost_v[0],
       nxtot = p->n_all_x[0],
       nvtot = p->n_all_v[0];
  Real dx = p->dx[0], dv = p->dv[0],
       dt = p->time_step,vmin = p->vmin[0];

  thrust::device_vector<Real> f,f_avatar(p->n_whole);

  quakins::PhaseSpaceInitialization<uInt,Real,DIM_X,DIM_V,
                                    SingleMaxwell> ps_init(p);
  ps_init(f);

  quakins::Integrator<Real> integrate(nvtot,nx,
                                      p->vmin[0]-dv*nvbd,p->vmax[0]+dv*nvbd);
 
  thrust::device_vector<Real> dens_tot(nx), dens(nx), potn(nx), Efield(nx);
  
  integrate(f.begin()+nxbd*nvtot,dens.begin());

  quakins::PoissonSolver<uInt,Real,DIM_X,FFT> solvePoissonEq(p->n_main_x,p->xmin,p->xmax);
  solvePoissonEq(dens,potn,Efield);

  quakins::gizmos::SolveEfieldByIntegration<uInt,Real,1> solveEfield(nx,dx);
    
  std::ofstream p_os("potn.qout",std::ios::out);  
  std::ofstream e_os("E.qout",std::ios::out);  
  std::ofstream d_os("dens.qout",std::ios::out);  

  quakins::SplittingShift<uInt,Real,
    quakins::FluxBalanceMethod> fsSolverX({dx,dt,nx,nv,nxbd,nvbd,nxtot});

  quakins::SplittingShift<uInt,Real,
    quakins::FluxBalanceMethod> fsSolverV({dv,dt,nv,nx,nvbd,nxbd,nvtot});


  quakins::QuantumSplittingShift<uInt,Real,DIM_V> wignerSolver({p->hbar});


  thrust::host_vector<Real> v_coord(nv),_Efield(nx);

  thrust::transform(thrust::host,
                    thrust::make_counting_iterator((uInt)0),
                    thrust::make_counting_iterator(nv),
                    v_coord.begin(),
                    [vmin,dv](uInt idx) { return vmin+0.5*dv+idx*dv; });

  quakins::ReorderCopy<uInt,Real,2> rocopy_fwd({nvtot,nxtot},{1,0});
  quakins::ReorderCopy<uInt,Real,2> rocopy_bwd({nxtot,nvtot},{1,0});
  
  rocopy_fwd(f.begin(),f.end(),f_avatar.begin());
  
  quakins::Boundary<uInt,PeriodicBoundary> x_boundary(nx,nxbd,0,nv);
  fsSolverX.prepare(v_coord);
  wignerSolver.prepare(potn);
  std::ofstream fcout("ff.qout"); 

  std::puts("main loop starts...");
  for (int step=0; step<p->stop_at; step++){
    std::cout << step << std::endl;
    x_boundary(f_avatar.begin(),f_avatar.end(),'c');
    fsSolverX.advance(f_avatar);

    rocopy_bwd(f_avatar.begin(),f_avatar.end(),f.begin());
    integrate(f.begin()+nxbd*nvtot,dens.begin());

    using namespace thrust::placeholders;
    thrust::transform(dens.begin(),dens.end(),
                      thrust::make_constant_iterator(1.0),
                      dens_tot.begin(),_1-_2);
    solvePoissonEq(dens_tot, potn, Efield);
    solveEfield(dens_tot, Efield);
    thrust::copy(Efield.begin(),Efield.end(),_Efield.begin()); 

    wignerSolver.advance(f);
    fsSolverV.prepare(_Efield);
    fsSolverV.advance(f);

    rocopy_fwd(f.begin(),f.end(),f_avatar.begin());

    if (step%5==0) { 
      using namespace quakins::diagnosis;
      d_os << dens_tot << std::endl;
      p_os << potn << std::endl;
      e_os << Efield << std::endl;
    }
  } 
  // fcout << std::endl;


}
  


