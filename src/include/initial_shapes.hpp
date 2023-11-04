#pragma once 


#include "Parameters.hpp"

namespace quakins {

template <typename val_type, int dim,
          template<typename,int> typename ShapeFunctor>
struct shape_packet_traits;


#include "initial_shapes/SingleMaxwell.inl"
#include "initial_shapes/SingleFermiDirac.inl"
#include "initial_shapes/TwoMaxwellStream.inl"

}
