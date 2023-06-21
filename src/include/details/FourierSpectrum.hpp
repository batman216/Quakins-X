#pragma once 

#include <cufft.h>




namespace quakins {

namespace details {


template <typename idx_type,
          typename val_type,
          idx_type dim,
          idx_type xdim, idx_type vdim>
class FourierSpectrum {

 
  cufftHandle plan_fwd, plan_inv;

  thrust::device_vector<cufftComplex> chunk_buffer;

public:
  template <typename Parameters>
  FourierSpectrum(Parameters *p, val_type dt) {


  }
 
  template <typename itor_type, typename vitor_type>
  void advance(itor_type itor_begin, itor_type itor_end, 
               vitor_type v_begin, int gpu) {
    

  }

};

} // namespace details

} // namespace quakins 
