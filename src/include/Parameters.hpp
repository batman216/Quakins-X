#pragma once

#include <array>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <exception>
#include <numeric>
#include <algorithm>
#include "quakins_exceptions.hpp"
#include "quakins_macros.hpp"
#include "util.hpp"


namespace quakins {


template <typename idx_type, typename val_type, int dim_x, int dim_v>
struct Parameters {

  /// dim_para stands for the dimension that is parallelized,
  /// quakins supports parallelization in only one dimension.
  const int mpi_rank, mpi_size, dim_para;

  using idx_X_t = std::array<idx_type,dim_x>;
  using val_X_t = std::array<val_type,dim_x>;
  using idx_V_t = std::array<idx_type,dim_v>;
  using val_V_t = std::array<val_type,dim_v>;

  // ------- domain box --------
  idx_X_t n_main_x, n_ghost_x,
          n_main_x_loc,
          n_all_x, n_all_x_loc; 

  idx_V_t n_main_v, n_ghost_v,
          n_main_v_loc,
          n_all_v, n_all_v_loc; 

  val_X_t dx, Lx, xmin, xmax;
  val_V_t dv, Lv, vmin, vmax;

  idx_type nx_whole, nv_whole, n_whole, n_whole_loc;
  idx_type n_whole_main, n_whole_main_loc;
  // -----------------------------

  // --------- time box ----------
  idx_type stop_at;
  val_type time_factor,time_step;
  // -----------------------------

  // ------- quantum box ---------
  val_type hbar, degeneracy;
  // -----------------------------
  
  // ------- io box ---------
  idx_type small_file_intp, large_file_intp;
  std::unordered_map<std::string, LinuxCommand> commands; 
  // -----------------------------


  Parameters();
  Parameters(int,int,int);

  void initial();
  void readTimeBox(std::string);
  void readDomainBox(std::string);
  void readQuantumBox(std::string);
  void readIOBox(std::string);

};


#include "details/Parameters.inl"



} // namespace quakins
