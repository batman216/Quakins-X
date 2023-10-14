#pragma once

#include "Integrator.hpp"
#include <fstream>
#include <iostream>
#include <thrust/copy.h>

namespace quakins {

namespace diagnosis {


template <typename T,typename val_type>
std::ostream& operator<<(std::ostream& os, const T& obj) {

  thrust::copy(obj.begin(),obj.end(),
               std::ostream_iterator<val_type>(os,"\t"));
  return os;

}

} // namespace diagnosis


} // namespace quakins
