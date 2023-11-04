#pragma once
#include <vector>
#include <thrust/for_each.h>
#include <thrust/adjacent_difference.h>
#include <cmath>

/****************************
 *
 *   df/dt + u df/dx = 0
 *
 * **************************/

template <typename idx_type, typename val_type>
struct Packet_fbm; 


namespace quakins {

template <typename idx_type, typename val_type>
class FluxBalanceMethod {
  
  using Packet = Packet_fbm<idx_type,val_type>;
  Packet p;

  std::vector<val_type> alpha;
  std::vector<int> shift;
  thrust::device_vector<val_type> flux;

public:
  FluxBalanceMethod(Packet);

  template <typename Container>
  void prepare(Container&);

  template <typename Container>
  void advance(Container&);
  
};


template <typename idx_type, typename val_type,
          template<typename,typename> typename T>
struct packet_traits;

template <typename idx_type, typename val_type>
struct packet_traits<idx_type,val_type,FluxBalanceMethod> {
  typedef Packet_fbm<idx_type,val_type> name;
};

} // namespace quakins

#include "details/FluxBalanceMethod.inl"
