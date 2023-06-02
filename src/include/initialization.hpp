#pragma once

#include <stdexcept>
#include <limits>
#include <omp.h>
#include "util.hpp"


template <typename idx_type, typename val_type, idx_type dim>
struct Parameters {

  idx_type n_dev;
  
  idx_type time_step_total;

  // time interval
  val_type dt;

  // # mesh grid
  std::array<idx_type,dim> n;

  // # ghost mesh
  std::array<idx_type,dim> n_ghost;

  // # total in each direction (n_tot=n1+2*n_ghost)
  std::array<idx_type,dim> n_tot, n_tot_local;

  // # n_tot 
  idx_type n_1d_tot, n_1d_per_dev;

  // boundary of the phase space 
  std::array<val_type,dim> low_bound, up_bound;

  // interval between the phase space mesh grid nodes.
  std::array<val_type,dim> interval;

  // length of each direction in phase space
  std::array<val_type,dim> length;
  
  std::string x1_shape, x2_shape, v1_shape, v2_shape;
  // characteristic velocties
  val_type v1, v2, v3, v4;


};

namespace quakins {


template<typename idx_type, typename val_type, idx_type dim>
void init(Parameters<idx_type,val_type, dim>* p,int mpi_rank) {
  
  std::ifstream input_file("quakins.input");

  auto t_map = read_box(input_file, "time");
  
  assign(p->dt, "dt", t_map);
  assign(p->time_step_total, "step_total", t_map);

  std::cout << "This shot is about to run "
            << p->time_step_total << " steps." << std::endl;

  auto d_map = read_box(input_file, "domain");
  
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
  
  // n_tot = n + 2*n_nd:  total #grids = 2*#boundary grid + #real grid
  std::transform(p->n.begin(), p->n.end(), p->n_ghost.begin(), p->n_tot.begin(),
                [](idx_type a, idx_type b){ return a+2*b; });

  std::transform(p->low_bound.begin(), p->low_bound.end(), 
                p->up_bound.begin(), p->length.begin(),
                [](val_type a, val_type b){ return b-a; });

  std::transform(p->length.begin(),p->length.begin()+2,
                 p->n.begin(),p->interval.begin(),
                [](val_type a, idx_type b){ return a/(static_cast<val_type>(b)-1); });

  std::transform(p->length.begin()+2,p->length.end(),
                 p->n.begin()+2,p->interval.begin()+2,
                [](val_type a, idx_type b){ return a/static_cast<val_type>(b); });

  p->n_1d_tot = std::accumulate(p->n_tot.begin(),p->n_tot.end(),
                static_cast<idx_type>(1),std::multiplies<idx_type>());




  int temp; // todo: this is not very elegant.
  cudaGetDeviceCount(&temp);
  p->n_dev = static_cast<idx_type>(temp);

  std::copy(p->n_tot.begin(),p->n_tot.end(),p->n_tot_local.begin());
  p->n_tot_local[dim-1] /= p->n_dev;

  p->n_1d_per_dev = p->n_1d_tot / p->n_dev;

  std::copy(p->n_tot_local.begin(),p->n_tot_local.end(),std::ostream_iterator<idx_type>(std::cout," "));
  std::cout << p->n_1d_tot << std::endl;

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
  
    std::size_t mem_size = p->n_1d_tot*sizeof(val_type)/1048576; 
    std::cout << "The Wigner function costs " 
              << mem_size << "Mb of Memory, " 
              << mem_size/p->n_dev << "Mb per GPU." << std::endl;
  }
  
  auto v_map = read_box(input_file,"initial");

  assign(p->v1,"v1",v_map);
  assign(p->v2,"v2",v_map);
  assign(p->v3,"v3",v_map);
  assign(p->v4,"v4",v_map);

     
  p->x1_shape = v_map["x1_shape"];
  p->x2_shape = v_map["x2_shape"];
  p->v1_shape = v_map["v1_shape"];
  p->v2_shape = v_map["v2_shape"];


  input_file.close();

  // execption
  auto step_length0 = p->dt*p->up_bound[0]/p->interval[2]; 
  auto step_length1 = p->dt*p->up_bound[1]/p->interval[3]; 

  if (  (int)step_length0>p->n_ghost[2]-2 
      ||(int)step_length1>p->n_ghost[3]-2 )
    throw std::invalid_argument("步子迈太大！");

}

template<typename idx_type, typename val_type, idx_type dim>
void init(Parameters<idx_type,val_type, dim>* p) {

  init(p,0);
}


} // namespace quakins

