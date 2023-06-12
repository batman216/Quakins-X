#pragma once

#include "details/poisson_algorithms.hpp"

namespace quakins {

template <typename idx_type,
          typename val_type,
          idx_type dim,
          typename Policy = FFT<idx_type,val_type,dim>>
class PoissonSolver {

  Policy *policy;

  const std::array<idx_type,dim> n_dim;
  const std::array<val_type,2*dim> bound;

public:

  PoissonSolver(std::array<idx_type,dim> n_dim,
                std::array<val_type,2*dim> bound)
  : n_dim(n_dim), bound(bound) {

    policy = new Policy(n_dim, bound);

  }
  
  template <typename itor_type> __host__
  void operator()(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    policy->solve(in_begin, in_end, out_begin);
      
  }

};

} // namespace quakins
