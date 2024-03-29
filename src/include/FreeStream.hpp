#pragma once

#include "details/FluxBalanceCoordSpace.hpp"
#include "details/FluxBalanceVeloSpace.hpp"
#include "details/FourierSpectrum.hpp"
#include "details/FourierSpectrumVeloSpace.hpp"
#include "details/WignerTerm.hpp"


namespace quakins {

template <typename idx_type,typename val_type, int dim, 
          template<typename,typename,int> typename Policy>
class FreeStream {

  typedef Policy<idx_type,val_type,dim> Algorithm; 

  Algorithm *policy;

public:
  
  template<typename Parameters,typename ParallelCommunicator>
  FreeStream(Parameters *p, 
             ParallelCommunicator *para, 
             val_type dt, int xdim=0, int vdim=0) {

    policy = new Algorithm(p, para, dt, xdim,vdim);  
    
  }

  template <typename itor_type, typename vitor_type>
  void operator()(itor_type itor_begin, itor_type itor_end, 
                  vitor_type v_begin) {
    
    policy->advance(itor_begin, itor_end, v_begin);
  }

};

} // namespace quakins
