/**
 * @file      gizmos.hpp
 * @author    Tian-Xing Hu
 * @brief     something interesting
 * @date      2023.10.14
 */

#pragma once 

#include <thrust/scan.h>
#include <thrust/transform.h>

namespace quakins {

namespace gizmos {

/*!
 *
 * solve the electric field directly by integrate the 
 * density disturbution, more details see for 
 * [Feng&Sheng Physica Scripta 2015 "An approach to 
 * numerically solving the Poisson equation."]
 *
 */
template <typename idx_type, typename val_type, int dim>
class SolveEfieldByIntegration; 


template <typename idx_type, typename val_type>
class SolveEfieldByIntegration<idx_type,val_type,1> {

  const idx_type n; const val_type h, L; 

  thrust::device_vector<val_type> Ebuf;
public:
  SolveEfieldByIntegration(idx_type n, val_type h)
  : n(n), h(h), L(n*h) { Ebuf.resize(n); }

  template <typename Container>
  void operator()(Container& dens, Container& Efield){
    
    using thrust::make_transform_iterator;
    using namespace thrust::placeholders;

    val_type dx = h, Lx = L, nx = n;

    auto titor = make_transform_iterator(dens.begin(),
                                         [dx]__host__ __device__(val_type val)
                                         { return val*dx; });

    thrust::exclusive_scan(titor, titor+n, Efield.begin());
    thrust::inclusive_scan(titor, titor+n, Ebuf.begin());
    thrust::transform(Efield.begin(), Efield.end(), Ebuf.begin(),
                      Efield.begin(), (_1+_2)*0.5);
    val_type E0 = thrust::reduce(Efield.begin(),Efield.end());

    thrust::for_each(Efield.begin(),Efield.end(),
                     [E0,nx]__host__ __device__(val_type& val)
                     { val -= E0/nx; });

  }

};

template <typename idx_type, typename val_type>
class SolveEfieldByIntegration<idx_type,val_type,2> {};

template <typename idx_type, typename val_type>
class SolveEfieldByIntegration<idx_type,val_type,3> {};

} // namespace gizmos

} // namespace quakins
