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

    val_type r = sqrtf(pow(z[2],2)+pow(z[3]-100.,2));

    return   .5/(.2*M_PI)*
             std::exp(-std::pow(z[0],2)/.2)
           * (std::exp(-std::pow(z[1]-1,2)/.2)+std::exp(-std::pow(z[1]+1,2)/.2))
           * (1+0.0001*std::exp(-pow(z[2]-30,2)/4)*cos(M_PI*0.05*z[3]));
          // * (1+0.001*cos(M_PI*0.05*z[3]));
           //* (1+0.01*cos(M_PI*0.1*z[2])+0.02*cos(M_PI*0.1*z[3]));
           //* (1.+std::exp(-std::pow(z[2]-30,2))
           //* std::exp(-std::pow(z[3]-20,2)));
          // * (std::exp(0.1*std::exp(-r)/r))
           //* j0f(3.6825*z[2]);

  }

};

