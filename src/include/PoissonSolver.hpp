#pragma once

#include "details/poisson_algorithms.hpp"

namespace quakins {

template <typename idx_type,
          typename val_type,
          idx_type dim,
          template<typename, typename,idx_type> typename Policy>
class PoissonSolver {

  Policy<idx_type,val_type,dim> *policy;

public:

  PoissonSolver(std::array<idx_type,dim> n_dim,
                std::array<val_type,2*dim> bound, char coord='d') {

    policy = new Policy<idx_type,val_type,dim>(n_dim, bound,coord);

  }
  ~PoissonSolver() { delete policy; }

  template <typename itor_type> 
  void operator()(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    policy->solve(in_begin, in_end, out_begin);
      
  }

};

} // namespace quakins
