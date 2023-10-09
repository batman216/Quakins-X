#pragma once 

#include <tuple>
#include <fstream>
#include <nvfunctional>
#include "../macros.hpp"
#include "../util.hpp"


template <typename idx_type, 
          typename val_type, 
          idx_type dim, typename Parameters>
struct XSpace {

  typedef std::array<val_type,dim> val_array;
  val_type operator()(Parameters *p, val_array z) {

  }

};


template <typename idx_type, 
          typename val_type, 
          idx_type dim, typename Parameters>
struct VSpace {

  typedef std::array<val_type,dim> val_array;
  val_type operator()(Parameters *p, val_array z) {
     
  }

};



template <typename val_type>
struct atonum;

template<>
struct atonum<int> {
  static int convert(std::string str) {
    return std::stoi(str);
  }
};

template<>
struct atonum<float> {
  static float convert(std::string str) {
    return std::stof(str);
  }
};

template<>
struct atonum<double> {
  static double convert(std::string str) {
    return std::stod(str);
  }
};

template <typename val_type>
struct readFromHost2Device {
  
  __host__ __device__
  val_type operator()(std::string str) {
  
    return atonum<val_type>::convert(str);
  }

};

template <typename idx_type, 
          typename val_type, 
          idx_type dim, typename Parameters>
struct ShapeFunctor { 

  typedef std::array<val_type,dim> val_array;

  // spatial parameters
  
  ShapeFunctor(Parameters *p) { }
  __host__ __device__
  val_type write(const val_array& z) {

    val_type vth = 0.4;

    auto f_m = []__host__ __device__ (val_type v1,  val_type v2,
                                      val_type vd1, val_type vd2,
                                      val_type vt1, val_type vt2){
      return expf(-(powf((v1-vd1)/vt1,2)+powf((v2-vd2)/vt2,2))/2)/(2.*M_PI); 
    };

    auto f_f = []__host__ __device__ (val_type v1,  val_type v2,
                                      val_type vd1, val_type vd2,
                                      val_type vt1, val_type vt2){
      return expf(-(powf((v1-vd1)/vt1,2)+powf((v2-vd2)/vt2,2))/2)/(2.*M_PI); 
    };

    return  
       // .5/(2*M_PI)*(expf(-powf(z[0]-.2,2)/vth/vth/2) 
        //            +expf(-powf(z[0]+.2,2)/vth/vth/2)) 
           .5*(f_m(z[0],z[1],0, 0.2,vth,vth)
              +f_m(z[0],z[1],0,-0.2,vth,vth))
           * (1.0+0.0001*cosf(M_PI/40.*2.*z[3]));
          // * (1+0.001*cos(M_PI*0.05*z[3]));
           //* (1+0.01*cos(M_PI*0.1*z[2])+0.02*cos(M_PI*0.1*z[3]));
           //* (1.+std::exp(-std::pow(z[2]-30,2))
          // * (std::exp(0.1*std::exp(-r)/r))
           //* j0f(3.6825*z[2]);

  }

};

