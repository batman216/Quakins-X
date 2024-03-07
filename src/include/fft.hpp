/**
 * @file      fft.hpp 
 * @brief     A fft API based on cufft
 * @author    Tian-Xing Hu
 * @copyright Tian-Xing Hu 2023
 */

#pragma once 
#include <cufft.h>
#include "fft_traits.hpp"

namespace quakins {


void fft_exec_forward(cufftHandle &plan, cufftReal* in, cufftComplex* out) {
    CUFFT_CALL( cufftExecR2C(plan,in,out) );
}

void fft_exec_forward(cufftHandle &plan, cufftDoubleReal* in, cufftDoubleComplex* out) {
    CUFFT_CALL( cufftExecD2Z(plan,in,out) );
}

void fft_exec_forward(cufftHandle &plan, cufftComplex* in, cufftComplex* out) {
    CUFFT_CALL( cufftExecC2C(plan,in,out,CUFFT_FORWARD) );
}

void fft_exec_forward(cufftHandle &plan, cufftDoubleComplex* in, cufftDoubleComplex* out) {
    CUFFT_CALL( cufftExecZ2Z(plan,in,out,CUFFT_FORWARD) );
}

void fft_exec_inverse(cufftHandle &plan, cufftComplex* in, cufftReal* out) {
    CUFFT_CALL( cufftExecC2R(plan,in,out) );
}

void fft_exec_inverse(cufftHandle &plan, cufftDoubleComplex* in, cufftDoubleReal* out) {
    CUFFT_CALL( cufftExecZ2D(plan,in,out) );
}

void fft_exec_inverse(cufftHandle &plan, cufftComplex* in, cufftComplex* out) {
    CUFFT_CALL( cufftExecC2C(plan,in,out,CUFFT_INVERSE) );
}

void fft_exec_inverse(cufftHandle &plan, cufftDoubleComplex* in, cufftDoubleComplex* out) {
    CUFFT_CALL( cufftExecZ2Z(plan,in,out,CUFFT_INVERSE) );
}

template <int fft_rank>
struct fft_many_args {
  std::array<int,fft_rank> n;
  std::array<int,fft_rank> inembed; int istride, idist;
  std::array<int,fft_rank> onembed; int ostride, odist; 
  int batch;
};

template <typename idx_type,
          typename val_type, int fft_rank>
class FFT;


template <typename idx_type, typename val_type>
class FFT<idx_type,val_type,1> {

  FFT_traits<val_type> fft_type;

  cufftHandle plan_fwd, plan_inv;

  using first_t = fft_pointer_traits<val_type>::first;
  using second_t = fft_pointer_traits<val_type>::second;

  idx_type n, n_batch, n_bd, n_tot;

public:
  FFT(idx_type n, idx_type n_batch, idx_type n_bd) 
  : n(n), n_batch(n_batch), n_bd(n_bd), n_tot(n+2*n_bd) {
   
    int nfft[1] = {static_cast<int>(n)};
    int ntot[1] = {static_cast<int>(n_tot)};
    
    CUFFT_CALL(cufftPlanMany(&plan_fwd, 1, nfft, 
                             ntot, 1, n_tot,
                             ntot, 1, n_tot,
                             fft_type.forward, n_batch));
    CUFFT_CALL(cufftPlanMany(&plan_inv, 1, nfft, 
                             ntot, 1, n_tot,
                             ntot, 1, n_tot,
                             fft_type.inverse, n_batch));
  }

  FFT(fft_many_args<1> fmany, fft_many_args<1> imany) {
   
    CUFFT_CALL(cufftPlanMany(&plan_fwd, 1, fmany.n.data(), 
                             fmany.inembed.data(), fmany.istride, fmany.idist,
                             fmany.onembed.data(), fmany.ostride, fmany.odist,
                             fft_type.forward, fmany.batch));

    CUFFT_CALL(cufftPlanMany(&plan_inv, 1, imany.n.data(), 
                             imany.inembed.data(), imany.istride, imany.idist,
                             imany.onembed.data(), imany.ostride, imany.odist,
                             fft_type.inverse, imany.batch));

  }

  ~FFT() { cufftDestroy(plan_fwd); cufftDestroy(plan_inv); }

  void forward(first_t *in_ptr, second_t *out_ptr) {
    fft_exec_forward(plan_fwd,in_ptr,out_ptr);
  }

  void inverse(second_t *in_ptr, first_t *out_ptr) {
    fft_exec_inverse(plan_inv,in_ptr,out_ptr);
  }


  template <typename itor_type>
  void forward(itor_type in_begin, itor_type out_begin) {

    auto in_ptr = reinterpret_cast<first_t*>(
                        thrust::raw_pointer_cast(&(*in_begin)));
    auto out_ptr = reinterpret_cast<second_t*>(
                        thrust::raw_pointer_cast(&(*out_begin)));
 
    this->forward(in_ptr,out_ptr);
  }

  template <typename itor_type>
  void inverse(itor_type in_begin, itor_type out_begin) {

    auto in_ptr = reinterpret_cast<second_t*>(
                        thrust::raw_pointer_cast(&(*out_begin)));
    auto out_ptr = reinterpret_cast<first_t*>(
                        thrust::raw_pointer_cast(&(*in_begin)));
    this->inverse(in_ptr,out_ptr);

  }

};


template <typename idx_type, typename val_type>
class FFT<idx_type,val_type,2> {

  cufftHandle plan_fwd, plan_inv;
  val_type norm;

  using idx_array = std::array<idx_type,2>;

public:
  FFT(idx_array n, idx_type n_batch) {

      int nv[2] = {static_cast<int>(n[1]), static_cast<int>(n[0]-2)}; 
      int rnembed[2] = {static_cast<int>(n[1]), static_cast<int>(n[0])}; 
      int cnembed[2] = {static_cast<int>(n[1]), static_cast<int>(n[0]/2)}; 
 
      int rdist = static_cast<int>(n[0]*n[1]);
      int cdist = static_cast<int>(n[0]*n[1]/2);

      CUFFT_CALL(
      cufftPlanMany(&plan_fwd, 2, nv, rnembed, 1, rdist,
                                      cnembed, 1, cdist,
                                      CUFFT_R2C, n_batch));
      CUFFT_CALL(
      cufftPlanMany(&plan_inv, 2, nv, cnembed, 1, cdist,
                                      rnembed, 1, rdist,
                                      CUFFT_C2R, n_batch));
      norm = static_cast<val_type>((n[0]-2)*n[1]);

  }

  ~FFT() {}

  template <typename itor_type>
  void forward(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    auto n_tot = in_end - in_begin;
    auto real_ptr = reinterpret_cast<cufftReal*>(
                        thrust::raw_pointer_cast(&(*in_begin)));
    auto comp_ptr = reinterpret_cast<cufftComplex*>(
                        thrust::raw_pointer_cast(&(*out_begin)));
 
    val_type Norm = norm;
    CUFFT_CALL(cufftExecR2C(plan_fwd, real_ptr, comp_ptr));
    
    thrust::for_each(thrust::device,
                     comp_ptr, comp_ptr+n_tot/2,
                     [Norm] __host__ __device__ 
                     (cufftComplex& val) { val.x/=Norm; val.y/=Norm; });
  }

  template <typename itor_type>
  void inverse(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    auto real_ptr = reinterpret_cast<cufftReal*>(
                        thrust::raw_pointer_cast(&(*out_begin)));
    auto comp_ptr = reinterpret_cast<cufftComplex*>(
                        thrust::raw_pointer_cast(&(*in_begin)));
   
    CUFFT_CALL(cufftExecC2R(plan_inv, comp_ptr, real_ptr));

  }

};


}
