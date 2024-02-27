#pragma once

#include <stdexcept>
#include <limits>
#include <omp.h>
#include "util.hpp"
#include "macros.hpp"


template <typename idx_type, typename val_type, int dim>
struct Parameters {

  idx_type n_dev;
  
  idx_type time_step_total;

  // time interval
  val_type dt;

  // # mesh grid
  std::array<idx_type,dim> n, n_local;

  // # ghost mesh
  std::array<idx_type,dim> n_ghost;

  // # total in each direction (n_all=n1+2*n_ghost)
  std::array<idx_type,dim> n_all, n_all_local;

  // # n_all 
  idx_type n_1d_all, n_1d_per_dev;

  // boundary of the phase space 
  std::array<val_type,dim> low_bound, up_bound;

  // interval between the phase space mesh grid nodes.
  std::array<val_type,dim> interval;

  // length of each direction in phase space
  std::array<val_type,dim> length;

  // normalize Planck's constant
  val_type hbar,Theta;

  idx_type dens_print_intv;

  std::unordered_map<std::string,LinuxCommand> runtime_commands;
};

namespace quakins {


template<typename idx_type, typename val_type, int dim>
void init(Parameters<idx_type,val_type, dim>* p, int mpi_rank) {
  
  std::ifstream input_file(INPUT_FILE);

  p->runtime_commands = readRuntimeCommand(input_file);

  std::cout << p->runtime_commands["copytoc"].command <<  std::endl;
  
  input_file.close();
  input_file.open(INPUT_FILE);
  auto t_map = readBox(input_file, "time");
  int temp; // todo: this is not very elegant.
  cudaGetDeviceCount(&temp);
  p->n_dev = static_cast<idx_type>(temp);

  assign(p->dt, "dt", t_map);
  assign(p->time_step_total, "step_total", t_map);

  std::cout << "This shot is about to run "
            << p->time_step_total << " steps." << std::endl;
  input_file.close();
  input_file.open(INPUT_FILE);

  auto d_map = readBox(input_file, "domain");
  
  assign(p->n[0], "nv1", d_map);
  assign(p->n[1], "nv2", d_map);
  assign(p->n[2], "nx1", d_map);
  assign(p->n[3], "nx2", d_map);
  assign(p->n_ghost[0], "nv1_ghost", d_map);
  assign(p->n_ghost[1], "nv2_ghost", d_map);
  assign(p->n_ghost[2], "nx1_ghost", d_map);
  assign(p->n_ghost[3], "nx2_ghost", d_map);

  assign(p->low_bound[0], "v1min", d_map);
  assign(p->low_bound[1], "v2min", d_map);
  assign(p->low_bound[2], "x1min", d_map);
  assign(p->low_bound[3], "x2min", d_map);
  
  assign(p->up_bound[0], "v1max", d_map);
  assign(p->up_bound[1], "v2max", d_map);
  assign(p->up_bound[2], "x1max", d_map);
  assign(p->up_bound[3], "x2max", d_map);

  input_file.close();
  input_file.open(INPUT_FILE);

  auto dia_map = readBox(input_file, "diagnosis");
  assign(p->dens_print_intv, "dens_print_intv", dia_map);

  // simulation domain size 
  std::transform(p->low_bound.begin(), p->low_bound.end(), 
                 p->up_bound.begin(), p->length.begin(),
                 [](val_type a, val_type b){ return b-a; });

  std::transform(p->length.begin(),p->length.end(),
                 p->n.begin(),p->interval.begin(),
                 [](val_type a, idx_type b){ return a/(static_cast<val_type>(b)); });

  // n_all = n + 2*n_nd:  total #grids = 2*#boundary grid + #real grid
  std::transform(p->n.begin(), p->n.end(), p->n_ghost.begin(), p->n_all.begin(),
                [](idx_type a, idx_type b){ return a+2*b; });
  std::copy(p->n_all.begin(),p->n_all.end(),p->n_all_local.begin());
  p->n_all_local[dim-1] /= p->n_dev;

  // NCCL communination grid on the outermost dimension
  p->n_all[3] += p->n_ghost[3]*(p->n_dev-1)*2;
  p->n_all_local[3] = p->n_all[3]/p->n_dev;
    
  p->n_1d_all = std::accumulate(p->n_all.begin(),p->n_all.end(),
                static_cast<idx_type>(1),std::multiplies<idx_type>());
  p->n_1d_per_dev = p->n_1d_all / p->n_dev;

  std::size_t mem_size = p->n_1d_all*sizeof(val_type)/1048576; 
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, mem_size/p->n_dev*2.2);

  input_file.close();
  input_file.open(INPUT_FILE);

  auto quantum_map = readBox(input_file, "Quantum");
  assign(p->hbar, "hbar", quantum_map);
  assign(p->Theta, "Theta", quantum_map);
  
  input_file.close();

  if (mpi_rank==0) {
    std::cout << "Maxium of your int type: " 
              << std::numeric_limits<idx_type>::max() << std::endl;

    std::cout << "You have " << omp_get_num_procs() << " CPU hosts." << std::endl;
    std::cout << p->n_dev << " GPU devices are found: " << std::endl;
    for (int i=0; i<p->n_dev; i++) {
      cudaDeviceProp dev_prop;
      cudaGetDeviceProperties(&dev_prop,i);
      std::cout << "  " << i << ": " << dev_prop.name << std::endl;
    } // display the names of GPU devices
  
    std::cout << "The Wigner function costs " 
              << 2*mem_size << "Mb of Memory, " 
              << 2*mem_size/p->n_dev << "Mb per GPU." << std::endl;
  }
  
  // execption
  auto step_length0 = p->dt*p->up_bound[0]/p->interval[2]; 
  auto step_length1 = p->dt*p->up_bound[1]/p->interval[3]; 

  if (  (int)step_length0>p->n_ghost[2]-1 
      ||(int)step_length1>p->n_ghost[3]-1 )
    throw std::invalid_argument("步子迈太大！");

}

template<typename idx_type, typename val_type, int dim>
void init(Parameters<idx_type,val_type, dim>* p) { init(p,0); }


} // namespace quakins

