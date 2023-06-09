#pragma once



template<typename val_type, typename itor_type>
void removeBufferGrids(itor_type in_begin, itor_type in_end, itor_type out_begin) {
  


}


template<typename idx_type,
         typename val_type, idx_type dim>
class FFT {
  

public:
  FFT(std::array<idx_type,dim> n_dim,
      std::array<val_type,2*dim> bound) {}

  template <typename itor_type> __host__ __device__ 
  void solve(itor_type in_begin, itor_type in_end, itor_type out_begin) {

  }

};

namespace quakins {

template <typename idx_type,
          typename val_type,
          idx_type dim,
          typename Policy = FFT<idx_type,val_type,dim>>
class PoissonSolver {

  Policy *policy;

  const std::array<idx_type,dim> n_dim;
  const std::array<val_type,2*dim> bound;

public:

  PoissonSolver(std::array<idx_type,dim> n_dim,
                std::array<val_type,2*dim> bound)
  : n_dim(n_dim), bound(bound) {

    policy = new Policy(n_dim, bound);

  }
  
  template <typename itor_type> __host__
  void operator()(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    policy->solve(in_begin, in_end, out_begin);
      
  }

};

} // namespace quakins
