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

template <typename idx_type, typename r_type, typename c_type>
struct Packet_fbm; 


namespace quakins {

template <typename idx_type, typename r_type, typename c_type>
class FluxBalanceMethod {
  
  using Packet = Packet_fbm<idx_type,r_type,c_type>;
  Packet p;

  std::vector<r_type> alpha;
  std::vector<int> shift;
  thrust::device_vector<c_type> flux;

public:
  FluxBalanceMethod(Packet);

  template <typename Container>
  void prepare(Container&);

  template <typename Container>
  void advance(Container&);
  
};


template <typename idx_type, typename r_type, typename c_type,
          template<typename,typename,typename> typename T>
struct packet_traits;

template <typename idx_type, typename r_type, typename c_type>
struct packet_traits<idx_type,r_type, c_type,FluxBalanceMethod> {
  typedef Packet_fbm<idx_type,r_type,c_type> name;
};

} // namespace quakins

#include "details/FluxBalanceMethod.inl"
