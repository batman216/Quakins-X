#pragma once 
#include <fstream>
#include <string>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

namespace quakins {
template <typename idx_type,
          typename val_type,
          idx_type dim, idx_type s1, idx_type s2>
class Slicer {

  std::ofstream out;
  idx_type n_tot;
  using idx_array = std::array<idx_type,dim>;
  idx_array shift,n;
  thrust::device_vector<val_type> out_con;
public:
  Slicer(idx_array n,int mpi_rank,std::string name): n(n) {

    out.open(name+"@"+std::to_string(mpi_rank),std::ios::out); 
    n_tot = n[s1]*n[s2];
    out_con.resize(n_tot);

    thrust::exclusive_scan(n.begin(),n.end(),shift.begin(),1,thrust::multiplies<val_type>());

  }
  template <typename itor_type>
  void operator()(idx_array loc, itor_type in_begin) {

    idx_type start = thrust::inner_product(loc.begin(),loc.end(),
                                           shift.begin(),0);

    idx_type n1 = n[s1];

    idx_type sh1 = shift[s1];
    idx_type sh2 = shift[s2];

    auto d_map = thrust::make_transform_iterator(
            thrust::make_counting_iterator(static_cast<idx_type>(0)),
              [sh1,sh2,n1]__host__ __device__(idx_type idx) {
                return static_cast<idx_type>(idx/n1*sh1 + idx%n1*sh2);
              });

    thrust::gather(thrust::device,d_map,d_map+n_tot,
                   in_begin+start, out_con.begin());
    
    thrust::copy(out_con.begin(),out_con.begin()+n_tot,std::ostream_iterator<val_type>(out," "));
    out << std::endl;

  }

  ~Slicer() { out.close(); }
};
}
