/**
 * @file      QuantumElectrostaticCoupling.hpp
 * @author    Tian-Xing Hu
 * @brief     Solving the mean-field term of the electrostatic Wigner equation
 * @date      2023.10.14
 */
#pragma once 
#include <thrust/device_vector.h>
#include "fft.hpp"

template <typename idx_type, typename val_type,int dim>
struct Packet_quantum;

template <typename idx_type, typename val_type>
struct Packet_quantum<idx_type,val_type,1>;
 

namespace quakins {

template <typename idx_type, typename val_type, int dim>
class QuantumSplittingShift;


template <typename idx_type, typename val_type>
class QuantumSplittingShift<idx_type,val_type,1>{
  

  using Packet = Packet_quantum<idx_type,val_type,1>;
  Packet p;

  /// 构造函数里先算好这些参数
  val_type qDt, qDl, Dl;

  thrust::device_vector<val_type> lambda, phase, hypercollision;

  FFT<idx_type,val_type,1> *fft;

  cudaArray_t phi_tex;
  cudaTextureObject_t tex_obj;
  
public:
  QuantumSplittingShift(Packet);

  template <typename Container>
  void prepare(Container&);
  template <typename Container> 
  void advance(Container&,Container&);

};

#include "details/QuantumSplittingShift.inl"

} // namespace  quakins
