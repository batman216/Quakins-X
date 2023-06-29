#pragma once

#include "details/FluxBalanceCoordSpace.hpp"
#include "details/FluxBalanceVeloSpace.hpp"
#include "details/FourierSpectrum.hpp"
#include "details/FourierSpectrumVeloSpace.hpp"
#include "details/WignerTerm.hpp"

template <typename idx_type, typename val_type, idx_type dim>
struct Parameters;

namespace quakins {

template <typename idx_type,
          typename val_type,
          idx_type dim, idx_type xdim, idx_type vdim, 
          template<typename,typename,idx_type,idx_type,
                   idx_type> typename Policy>
class FreeStream {

  typedef Policy<idx_type,val_type,dim,xdim,vdim> Algorithm; 

  Algorithm *policy;
public:
  
  template<typename Parameters>
  FreeStream(Parameters *p, val_type dt) {

    policy = new Algorithm(p, dt);  
    
  }

  template <typename itor_type, typename vitor_type>
  void operator()(itor_type itor_begin, itor_type itor_end, 
                  vitor_type v_begin, int gpu) {
    
    policy->advance(itor_begin, itor_end, v_begin, gpu);
  }

};

} // namespace quakins
