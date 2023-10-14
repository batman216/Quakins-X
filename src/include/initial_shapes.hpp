#pragma once 

#include <tuple>
#include <fstream>
#include <nvfunctional>



template <typename val_type, int dim>
struct SingleMaxwell; 

template <typename val_type, int dim>
struct TwoStream; 

template <typename val_type>
struct SingleMaxwell<val_type,2> { 

  
  using val_XV_t = std::array<val_type,2>;
  __host__ __device__
  static val_type shape(const val_XV_t& z) {

    val_type vth = 1, ptb = 0.01;

    auto f = []__host__ __device__ (val_type v1,
                                    val_type vd1,
                                    val_type vt1) {
      return expf(-powf((v1-vd1)/vt1,2)/2)/sqrtf(2.*M_PI)/vt1; 
    };

    return f(z[0],0,vth) * (1.0+ptb*cosf(M_PI/20.*2.*z[1]));

  }

};

template <typename val_type>
struct TwoStream<val_type,4> { 

  
  using val_XV_t = std::array<val_type,4>;
  __host__ __device__
  static val_type shape(const val_XV_t& z) {

    val_type vth = 0.4, ptb = 0.0001, vd = 2;


    auto f = []__host__ __device__ (val_type v1,  val_type v2,
                                      val_type vd1, val_type vd2,
                                      val_type vt1, val_type vt2){
      return expf(-(powf((v1-vd1)/vt1,2)+powf((v2-vd2)/vt2,2))/2)/(2.*M_PI*vt1*vt2); 
    };

    return .5*(f(z[0],z[1],0, vd/2,vth,vth)
              +f(z[0],z[1],0,-vd/2,vth,vth))
           * (1.0+ptb*cosf(M_PI/40.*2.*z[3]));

  }

};

template <typename val_type>
struct TwoStream<val_type,2> { 
  
  using val_XV_t = std::array<val_type,2>;
  __host__ __device__
  static val_type shape(const val_XV_t& z) {

    val_type vth = 0.4, vd = 2.0, ptb = 0.1, k0=M_PI/50*2;

    auto f = []__host__ __device__ (val_type v1,
                                    val_type vd1,
                                    val_type vt1) {
      return expf(-powf((v1-vd1)/vt1,2)/2)/sqrtf(2.*M_PI*vt1); 
    };

    return .5*(f(z[0], vd/2, vth)+f(z[0],-vd/2,vth))
             *(1.0+ptb*cosf(k0*z[1]));

  }

};

