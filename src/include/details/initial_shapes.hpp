#pragma once 

#include <concepts>
#include <tuple>




template <typename idx_type, 
          typename val_type, 
          idx_type dim, typename Parameters>
struct ShapeFunctor { 

  typedef std::array<val_type,dim> val_array;
  ShapeFunctor(Parameters *p) {}

  __host__ __device__
  val_type write(const val_array& z) {

    val_type r = sqrtf(pow(z[2],2)+pow(z[3]-30.,2));

    return   1./(1.*M_PI)*
             std::exp(-std::pow(z[0],2)/1)
           * std::exp(-std::pow(z[1],2)/1)
           * (1.+std::exp(-r)/r);
           //* (1+0.2*std::cos(2*M_PI*z[1]/12));
           //* j0f(3.6825*z[2]);

  }

};

