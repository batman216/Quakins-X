#pragma once

#include "FluxBalanceMethod.hpp"


namespace quakins {

/****************************
 *
 *   df/dt + u df/dx = 0
 *
 * **************************/


template <typename idx_type, typename val_type, 
          template<typename,typename> typename Algorithm>
class SplittingShift{

  Algorithm<idx_type,val_type> *algorithm;
  using Packet = packet_traits<idx_type,val_type,Algorithm>::name;

public:
  
  SplittingShift(Packet p) {
    algorithm = new Algorithm<idx_type,val_type>(p);
  }

  template <typename Container>
  void prepare(Container& con) { 
    algorithm->prepare(con);
  }

  template <typename Container>
  void advance(Container& con) { 
    algorithm->advance(con);
  }


};


} // namespace quakins
