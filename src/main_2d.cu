/**
 * @file      main_2d.cu
 * @brief     main file of the quakins 2d2v simualtion
 * @author    Tian-Xing Hu
 * @copyright Tian-Xing Hu 2023
 */


#include <iostream>
#include <mpi.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>

/// Quakins is highly modular, each main file is a dependent 
/// simualtion model based on the modules in the namespace quakins
#include "include/Parameters.hpp"
#include "include/PhaseSpaceInitialization.hpp"
#include "include/ParallelCommunicator.hpp"
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
    Parameters<uInt,Real,DIM_X,DIM_V>(mpi_rank, mpi_size, 1);
  p->initial(); // read from the input file.

  auto *para = new quakins::
    ParallelCommunicator<uInt,Real>(mpi_rank,mpi_size,MCW);

/// #1 preparation: copy the parameter in object p for convenience.
  uInt nx1 = p->n_main_x[0], nx2 = p->n_main_x[1];
  uInt nv1 = p->n_main_v[0], nv2 = p->n_main_v[1];

  uInt nx1bd = p->n_ghost_x[0], nx2bd = p->n_ghost_x[1],
       nv1bd = p->n_ghost_v[0], nv2bd = p->n_ghost_v[1];

  uInt nx1tot = p->n_all_x[0], nx2tot = p->n_all_x[1],
       nv1tot = p->n_all_v[0], nv2tot = p->n_all_v[1];
  uInt nxtot = nx1tot*nv2tot, nvtot = nv1tot*nv2tot;

  Real dx1 = p->dx[0], dx2 = p->dx[1];
  Real dv1 = p->dv[0], dv2 = p->dv[1];
  Real dt  = p->time_step;

/// #2 initialization: prepare the storge, initialize the simulation region.
  std::cout << p->n_whole_loc << std::endl;
  thrust::device_vector<Real> elec_f1(p->n_whole_loc/BLOCK),
                              elec_f2(p->n_whole_loc/BLOCK);
  thrust::host_vector<Real> _elec_block[BLOCK];
  quakins::PhaseSpaceInitialization<uInt,Real,DIM_X,DIM_V,
                                    quakins::SingleMaxwell> ps_init(p);
  ps_init(elec_f1);
 
  quakins::PhaseSpaceParallelCommute<uInt,Real> ps_nccl_com(nx1bd*nvtot,para);
  quakins::simple_print("df@"+std::to_string(mpi_rank),elec_f1); 

  /// Real space fields, no parallelization
  /// vector fields
  std::array<quakins::device_vector<Real>,3> 
    elec_field, magn_field, magn_pote;
  /// scalar fields
  quakins::device_vector<Real> elec_pote(nxtot),
             elct_dens(nxtot), ions_dens(nxtot),
             chge_dens(nxtot), current(nxtot);


/// #3 prepare the functors




/// #4 simulation main loop



/// #4
}
 
