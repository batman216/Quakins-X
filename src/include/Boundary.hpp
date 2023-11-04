#pragma once

#include "details/boundary_condition.hpp"

namespace quakins {

template <typename idx_type, 
          template<typename> typename Policy>
class Boundary {

  Policy<idx_type> *policy;

public:
  template <typename ...Ts>
  Boundary(Ts&& ...ts) { policy = new Policy<idx_type>(std::forward<Ts>(ts)...); }

  template<typename itor_type> 
  void operator()(itor_type itor_begin, itor_type itor_end) 
  { policy->implement(itor_begin,itor_end); }
  
};          



} // namespace quakins
