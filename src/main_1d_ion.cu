/**
 * @file      main_1d_ion.cu
 * @brief     main file of the quakins 1d1v simualtion with ions
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
template<typename Real,int dim> 
using init_shape = quakins::TwoStream<Real,dim>;

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
  
  quakins::Probe diag(p);

  uInt nx = p->n_main_x[0], nv = p->n_main_v[0], 
       nxbd = p->n_ghost_x[0],
       nvbd = p->n_ghost_v[0],
       nxtot = p->n_all_x[0],
       nvtot = p->n_all_v[0];
  Real dx = p->dx[0], dv = p->dv[0],
       dt = p->time_step,vmin = p->vmin[0],
       Lx = p->xmax[0]-p->xmin[0],
       Lv = p->vmax[0]-p->vmin[0]+2*dv*nvbd;

  thrust::device_vector<Real> fe,fe_avatar(p->n_whole);
  thrust::device_vector<Real> fi,fi_avatar(p->n_whole);

  quakins::PhaseSpaceInitialization<uInt,Real,DIM_X,DIM_V,
                                    quakins::TwoStream> ps_init_e(p);
  quakins::PhaseSpaceInitialization<uInt,Real,DIM_X,DIM_V,
                                    quakins::SingleMaxwell> ps_init_i(p);
  ps_init_e(fe);  ps_init_i(fi);

  quakins::Integrator<Real> integrate(nvtot,nx,
                                      p->vmin[0]-dv*nvbd,p->vmax[0]+dv*nvbd);
 
  thrust::device_vector<Real> dens_tot(nx), dense(nx), densi(nx), potn(nx), Efield(nx);
  
  integrate(fe.begin()+nxbd*nvtot,dense.begin());
  integrate(fi.begin()+nxbd*nvtot,densi.begin());

  quakins::PoissonSolver<uInt,Real,DIM_X,FFT> solvePoissonEq(p->n_main_x,p->xmin,p->xmax);

  quakins::SplittingShift<uInt,Real,
    quakins::FluxBalanceMethod> fsSolverX({dx,dt,nx,nv,nxbd,nvbd,nxtot});

  quakins::SplittingShift<uInt,Real,
    quakins::FluxBalanceMethod> fsSolverV({dv,dt,nv,nx,nvbd,nxbd,nvtot});

  std::cout << "hbar->" << p->hbar << std::endl;
  quakins::QuantumSplittingShift<uInt,Real,DIM_V> 
    quantumSolver({p->hbar,p->time_step, Lv,Lx,nv,nvbd,nx,nxbd,nvtot});


  thrust::host_vector<Real> v_coord(nv), _ae(nx), _ai(nx);

  thrust::transform(thrust::host,
                    thrust::make_counting_iterator((uInt)0),
                    thrust::make_counting_iterator(nv),
                    v_coord.begin(),
                    [vmin,dv](uInt idx) { return vmin+0.5*dv+idx*dv; });

  quakins::ReorderCopy<uInt,Real,2> rocopy_fwd({nvtot,nxtot},{1,0});
  quakins::ReorderCopy<uInt,Real,2> rocopy_bwd({nxtot,nvtot},{1,0});
  
  rocopy_fwd(fi.begin(),fi.end(),fi_avatar.begin());
  rocopy_fwd(fe.begin(),fe.end(),fe_avatar.begin());
  
  quakins::Boundary<uInt,PeriodicBoundary> x_boundary(nx,nxbd,0,nv);
  fsSolverX.prepare(v_coord);

  std::puts("main loop starts...");
  for (int step=0; step<p->stop_at; step++){

    std::cout << step << std::endl;

    /// set boundary condition for the x direction
    x_boundary(fe_avatar.begin(),fe_avatar.end(),'c');
    x_boundary(fi_avatar.begin(),fi_avatar.end(),'c');
    
    /// advance the x direction
    fsSolverX.advance(fe_avatar);
    fsSolverX.advance(fi_avatar);

    /// reorder the distribution from x-major to v-major
    rocopy_bwd(fe_avatar.begin(),fe_avatar.end(),fe.begin());
    rocopy_bwd(fi_avatar.begin(),fi_avatar.end(),fi.begin());

    /// calculate the density by integration
    integrate(fe.begin()+nxbd*nvtot,dense.begin());
    integrate(fi.begin()+nxbd*nvtot,densi.begin());

    /// sum up densities of different species
    using namespace thrust::placeholders;
    thrust::transform(dense.begin(),dense.end(), densi.begin(),
                      dens_tot.begin(),_1-_2);


    /// solve Poisson equation to obtain potential (and E-field if needed)
    solvePoissonEq(dens_tot, potn, Efield);
    //solveEfield(dens_tot, Efield);

    thrust::copy(Efield.begin(),Efield.end(),_ae.begin()); 
    thrust::copy(Efield.begin(),Efield.end(),_ai.begin()); 
    thrust::for_each(_ae.begin(),_ae.end(),[](Real& val){ val=-val; }); 
    thrust::for_each(_ai.begin(),_ai.end(),[](Real& val){ val/=1836.15; }); 

    fsSolverV.prepare(_ae);
    fsSolverV.advance(fe);
    fsSolverV.prepare(_ai);
    fsSolverV.advance(fi);
    
    rocopy_fwd(fi.begin(),fi.end(),fi_avatar.begin());
    rocopy_fwd(fe.begin(),fe.end(),fe_avatar.begin());

    diag.print(step,fe,dense,potn,Efield);
  } 
  // fcout << std::endl;


}
  


