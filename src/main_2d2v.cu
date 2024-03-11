/**
 * @file      main_2d.cu
 * @brief     main file of the quakins 2d2v simualtion
 * @author    Tian-Xing Hu
 * @copyright Tian-Xing Hu 2023
 */


#include "include/quakins_headers.hpp"

using Nums = std::size_t;
using Real = float;
using Complex = thrust::complex<Real>;

#define DIM_X 2
#define DIM_V 2
#define BLOCK 1

int main(int argc, char* argv[]) {

  int mpi_size, mpi_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MCW,&mpi_size);
  MPI_Comm_rank(MCW,&mpi_rank);

  auto *p = new quakins:: // parallelized space dimension ↓
    Parameters<Nums,Real,DIM_X,DIM_V>(mpi_rank, mpi_size, 1);
  p->initial(); // read from the input file.

  auto *para = new quakins::
    ParallelCommunicator<Nums,Real>(mpi_rank,mpi_size,MCW);

/// #1 preparation: copy the parameter in object p for convenience.
  Nums nx1 = p->n_main_x[0], nx2 = p->n_main_x[1], nx2loc = nx2/mpi_size,
       nv1 = p->n_main_v[0], nv2 = p->n_main_v[1];

  Nums nx1bd = p->n_ghost_x[0], nx2bd = p->n_ghost_x[1],
       nv1bd = p->n_ghost_v[0], nv2bd = p->n_ghost_v[1];

  /// 'tot': # including the boundary
  Nums nx1tot = p->n_all_x[0], nx2tot = p->n_all_x[1], nx2totloc = nx2tot/mpi_size,
       nv1tot = p->n_all_v[0], nv2tot = p->n_all_v[1];
  Nums nxtot = nx1tot*nx2tot, nvtot = nv1tot*nv2tot, ntot = nxtot*nvtot,
       ntotloc = ntot/mpi_size, nxtotloc = nxtot/mpi_size;

  Real dx1 = p->dx[0], dx2 = p->dx[1], 
       dv1 = p->dv[0], dv2 = p->dv[1];
  Real dt  = p->time_step;
  
  Real v1min = p->vmin[0], v2min = p->vmin[1];
  Real v1max = p->vmax[0], v2max = p->vmax[1];

/// #2 initialization: prepare the storge, initialize the simulation region.
  thrust::device_vector<Real> elec_f1(ntotloc/BLOCK),
                              elec_f2(ntotloc/BLOCK);
  thrust::host_vector<Real> _elec_block[BLOCK];
  quakins::PhaseSpaceInitialization<Nums,Real,DIM_X,DIM_V,
                                    quakins::TestShape> ps_init(p);
  ps_init(elec_f1);
 

  /// Real space fields, no parallelization
  /// vector fields
  std::array<thrust::device_vector<Real>,3> 
    elec_field, magn_field, magn_pote;
  /// scalar fields
  thrust::device_vector<Real> elec_pote(nxtotloc),
            elct_dens(nxtotloc), ions_dens(nxtotloc),
            chge_dens(nxtotloc), current(nxtotloc);


/// #3 prepare the quakins functors
  
  /// double integral functors  
  thrust::device_vector<Real> intg_buff(ntotloc/nv1tot);
  quakins::Integrator<Real> 
    integral1(nv1tot,ntotloc/nv1tot,v1min,v1max);
  quakins::Integrator<Real> 
    integral2(nv2tot,nxtotloc, v2min,v2max);
 
  integral1(elec_f1.begin(),intg_buff.begin());
  integral2(intg_buff.begin(),elct_dens.begin());

  /// nccl communication functor
  quakins::PhaseSpaceParallelCommute<Nums,Real> 
    ps_nccl_comm(nx2bd*nx1tot*nvtot,para);

  /// reorder copy functors
  quakins::PermutationCopy<Nums,Real,4> 
    pcopy0({nv1tot,nv2tot,nx1tot,nx2totloc},{3,2,0,1});

  quakins::PermutationCopy<Nums,Real,4> 
    pcopy1({nx2totloc,nx1tot,nv1tot,nv2tot},{1,0,3,2});
  quakins::PermutationCopy<Nums,Real,4> 
    pcopy2({nx1tot,nx2totloc,nv2tot,nv1tot},{3,2,0,1});
  quakins::PermutationCopy<Nums,Real,4> 
    pcopy3({nv1tot,nv2tot,nx1tot,nx2totloc},{1,0,2,3});
  quakins::PermutationCopy<Nums,Real,4> 
    pcopy4({nv2tot,nv1tot,nx1tot,nx2totloc},{3,2,1,0});

  /// stream solver functors
  quakins::SplittingShift<Nums,Real,Real,quakins::FluxBalanceMethod>
    fsSolverX1({mpi_rank,mpi_size,dx1,dt/2,nx1,nv1,nx1bd,nv1bd,ntotloc/nv1tot});
  quakins::SplittingShift<Nums,Real,Real,quakins::FluxBalanceMethod>
    fsSolverX2({mpi_rank,mpi_size,dx2,dt/2,nx2loc,nv2,nx2bd,nv2bd,ntotloc/nv2tot});

  quakins::SplittingShift<Nums,Real,Real,quakins::FluxBalanceMethod>
    fsSolverV1({mpi_rank,mpi_size,dv1,dt,nv1,nx1,nv1bd,nx1bd,ntot/nxtot});
  quakins::SplittingShift<Nums,Real,Real,quakins::FluxBalanceMethod>
    fsSolverV2({mpi_rank,mpi_size,dv2,dt,nv2,nx2loc,nv2bd,nx2bd,ntot/nxtot});

  thrust::host_vector<Real> v1(nv1), v2(nv2);
  thrust::transform(thrust::make_counting_iterator((Nums)0),
                    thrust::make_counting_iterator(nv1), v1.begin(),
                    [v1min,dv1]__host__ __device__(Nums idx) 
                    { return v1min+0.5*dv1+idx*dv1; });
  thrust::transform(thrust::make_counting_iterator((Nums)0),
                    thrust::make_counting_iterator(nv2), v2.begin(),
                    [v2min,dv2]__host__ __device__(Nums idx) 
                    { return v2min+0.5*dv2+idx*dv2; });

/// #4 simulation main loop
  unsigned int step{0};
  /// step 0
  fsSolverX1.prepare(v1);  fsSolverX2.prepare(v2);
  pcopy0(elec_f1.begin(),elec_f1.end(),elec_f2.begin());
  ps_nccl_comm(elec_f2.begin(),elec_f2.end());

  /// step 1 to end
  while (step++<p->stop_at) {

    std::cout << step << std::endl;

    fsSolverX2.advance(elec_f2);
    pcopy1(elec_f2.begin(),elec_f2.end(),elec_f1.begin());
    fsSolverX1.advance(elec_f1);
    pcopy2(elec_f1.begin(),elec_f1.end(),elec_f2.begin());
    ps_nccl_comm(elec_f2.begin(),elec_f2.end());

    if (step==p->stop_at) {
    integral1(elec_f2.begin(),intg_buff.begin());
    integral2(intg_buff.begin(),elct_dens.begin());
    quakins::simple_print("edens@"+std::to_string(mpi_rank),elct_dens); 
    }

    fsSolverV1.advance(elec_f2);
    pcopy3(elec_f2.begin(),elec_f2.end(),elec_f1.begin());
    fsSolverV2.advance(elec_f1);
    pcopy4(elec_f1.begin(),elec_f1.end(),elec_f2.begin());
  }
/// #4

}
 
