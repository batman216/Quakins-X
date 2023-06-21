#pragma once


#include <cuda_runtime.h>
#include <texture_types.h>

#include <cuda_texture_types.h>

namespace quakins {

namespace details {

template <typename idx_type,
          typename val_type,
          idx_type dim,
          idx_type xdim, idx_type vdim>
class WignerTerm {

public:
  template <typename Parameters>
  WignerTerm(Parameters *p, val_type dt) {}
  
  template <typename itor_type, typename vitor_type>
  void advance(itor_type itor_begin, itor_type itor_end, 
               vitor_type v_begin, int gpu) {}
 
};




} // details

} // quakins
