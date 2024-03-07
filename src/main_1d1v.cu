/**
 * @file      main_1d.cu
 * @brief     main file of the quakins 1d1v simualtion
 * @author    Tian-Xing Hu
 * @copyright Tian-Xing Hu 2023
 */

#include "include/quakins_headers.hpp"

using uInt = std::size_t;
using Real = float;
using Complex = thrust::complex<Real>;

#define DIM_X 1
#define DIM_V 1

int main(int argc, char* argv[]) {

  int mpi_size,  mpi_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MCW,&mpi_size);
  MPI_Comm_rank(MCW,&mpi_rank);

  auto *p = new quakins::
    Parameters<uInt,Real,DIM_X,DIM_V>(mpi_rank,mpi_size,0);

  auto *para = new quakins::
    ParallelCommunicator<uInt,Real>(mpi_rank,mpi_size,MCW);


  p->initial();
  

  uInt nx = p->n_main_x[0], nv = p->n_main_v[0], 
       nxloc = p->n_main_x_loc[0],
       nxbd = p->n_ghost_x[0], nvbd = p->n_ghost_v[0],
       nxtot = p->n_all_x[0],  nvtot = p->n_all_v[0],
       nxtotloc = p->n_all_x_loc[0];
  Real dx = p->dx[0], dv = p->dv[0],
       dt = p->time_step,vmin = p->vmin[0],
       Lx = p->xmax[0]-p->xmin[0],
       Lv = p->vmax[0]-p->vmin[0]+2*dv*nvbd;

  
  thrust::device_vector<Real> f,f_avatar(p->n_whole_loc+2*nxtotloc);

  quakins::PhaseSpaceInitialization<uInt,Real,DIM_X,DIM_V,
                                    quakins::TwoMaxwellStream> ps_init(p);
  ps_init(f);
   
  quakins::PhaseSpaceParallelCommute<uInt,Real> ps_nccl_com(nxbd*nvtot,para);
  
  quakins::Integrator<Real> integrate(nvtot,nxloc,
                                      p->vmin[0]-dv*nvbd,p->vmax[0]+dv*nvbd);
 
  thrust::device_vector<Real> dens_tot(nx), dens_e(nx),dens_e_loc(nxloc), potn(nx), Efield(nx);
  quakins::DensityAllGather<uInt,Real> dens_allgather(nxloc,para);

  integrate(f.begin()+nxbd*nvtot,dens_e_loc.begin());
  dens_allgather(dens_e.begin(),dens_e_loc.begin());

  quakins::PoissonSolver<uInt,Real,DIM_X,FFT> solvePoissonEq(p->n_main_x,p->xmin,p->xmax);
    
  quakins::SplittingShift<uInt,Real,Real,
    quakins::FluxBalanceMethod> fsSolverX({dx,dt/2,nxloc,nv,nxbd,nvbd,nxtotloc});

  quakins::SplittingShift<uInt,Real,Real,
    quakins::FluxBalanceMethod> fsSolverV({dv,dt,nv,nxloc,nvbd,nxbd,nvtot});

  quakins::QuantumSplittingShift<uInt,Real,DIM_V> 
    quantumSolver({p->hbar,p->time_step, 
                   Lv,Lx,nv,nvbd,nxloc,nxbd,nvtot,mpi_rank,mpi_size});

  thrust::host_vector<Real> v_coord(nv), _F(nxloc);

  thrust::transform(thrust::host,
                    thrust::make_counting_iterator((uInt)0),
                    thrust::make_counting_iterator(nv),
                    v_coord.begin(),
                    [vmin,dv](uInt idx) { return vmin+0.5*dv+idx*dv; });

  quakins::Probe diag(p);

  quakins::ReorderCopy<uInt,Real,2> rocopy_fwd({nvtot,nxtotloc},{1,0});
  quakins::ReorderCopy<uInt,Real,2> rocopy_bwd({nxtotloc,nvtot},{1,0});
  
  ps_nccl_com(f.begin(),f.end());
  rocopy_fwd(f.begin(),f.end(),f_avatar.begin());
  diag.print(0,f,dens_e,potn,Efield);
  
  quakins::Boundary<uInt,PeriodicBoundary> x_boundary(nx,nxbd);
  fsSolverX.prepare(v_coord);

  std::puts("main loop starts...");
  for (int step=1; step<p->stop_at; step++){

    __THE_FOLLOWING_CODE_ONLY_RUN_ON_RANK0__
    std::cout << step << std::endl;
    __THE_ABOVE_CODE_ONLY_RUN_ON_RANK0__

    /// set boundary condition for the x direction
    //x_boundary(f_avatar.begin(),f_avatar.end());
    
    /// advance the x direction
    fsSolverX.advance(f_avatar);

    /// reorder the distribution from x-major to v-major
    rocopy_bwd(f_avatar.begin(),f_avatar.end(),f.begin());

    // nccl communication should conduct when the data is v-major
    ps_nccl_com(f.begin(),f.end());

    /// calculate the density by integration
    integrate(f.begin()+nxbd*nvtot,dens_e_loc.begin());
    dens_allgather(dens_e.begin(),dens_e_loc.begin());
    
    /// sum up densities of different species
    using namespace thrust::placeholders;
    thrust::transform(dens_e.begin(),dens_e.end(),
                      thrust::make_constant_iterator(1.0),
                      dens_tot.begin(),_1-_2);

    /// solve Poisson equation to obtain potential (and E-field if needed)
    solvePoissonEq(dens_tot, potn, Efield);
    //solveEfield(dens_tot, Efield);

    if (p->hbar==0) { /// classical
      thrust::copy(Efield.begin()+mpi_rank*nxloc,Efield.begin()+(mpi_rank+1)*nxloc,_F.begin()); 
      thrust::for_each(_F.begin(),_F.end(),[](Real& val){ val=-val; }); 
      fsSolverV.prepare(_F);
      fsSolverV.advance(f);
    } else { /// quantum
      quantumSolver.prepare(potn);
      quantumSolver.advance(f,f_avatar);
    }
    
    rocopy_fwd(f.begin(),f.end(),f_avatar.begin());

    /// advance the x direction
    fsSolverX.advance(f_avatar);

    diag.print(step,f,dens_e,potn,Efield);
  } 

}

/// Valar morghuris
/// Valar dohaelis


