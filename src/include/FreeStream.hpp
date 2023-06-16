#pragma once

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

  template <typename itor_type>
  void operator()(itor_type itor_begin, itor_type itor_end, 
                idx_type n_chunk,int gpu) {
    
    policy->advance(itor_begin, itor_end, n_chunk, gpu);
  }

};

} // namespace quakins
