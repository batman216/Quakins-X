#pragma once

#include "details/poisson_algorithms.hpp"

namespace quakins {

template <typename idx_type,
          typename val_type,
          idx_type dim,
          template<typename, typename, int> typename Policy>
class PoissonSolver {

  Policy<idx_type,val_type,dim> *policy;

public:

  PoissonSolver(std::array<idx_type,dim> n_dim,
                std::array<val_type,dim> lef_bd,
                std::array<val_type,dim> rig_bd) {

    policy = new Policy<idx_type,val_type,dim>(n_dim, lef_bd, rig_bd);

  }
  ~PoissonSolver() { delete policy; }

  template <typename Container> // perfect forward
  void operator()(Container &&dens, Container &&potn) {
 
    policy->solve(dens,potn);
      
  }

  template <typename Container> // perfect forward
  void operator()(Container &&dens, Container &&potn, Container&& Efield) {
 
    policy->solve(dens,potn,Efield);
      
  }

};

} // namespace quakins
