#pragma once

#include "details/boundary_condition.hpp"

namespace quakins {

template <typename idx_type, 
          template<typename> typename Policy>
class Boundary {
  Policy<idx_type> *policy;

public:

  Boundary(idx_type nx, idx_type n_bd, 
                    idx_type n_step, idx_type n_chunk) { 
    policy = new Policy<idx_type>(nx,n_bd,n_step,n_chunk);
  }


  template<typename itor_type> __host__
  void operator()(itor_type itor_begin, itor_type itor_end,char flag) {

    policy->implement(itor_begin,itor_end,flag);
  }


};



} // namespace quakins
