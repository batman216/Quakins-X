#pragma once 
#include <cufft.h>

#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                    \
    {                                                                         \
        auto status = static_cast<cufftResult>( call );                       \
        if ( status != CUFFT_SUCCESS )                                        \
            fprintf( stderr,                                                  \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed " \
                     "with "                                                  \
                     "code (%d).\n",                                          \
                     #call,                                                   \
                     __LINE__,                                                \
                     __FILE__,                                                \
                     status );                                                \
    }
#endif  // CUFFT_CALL


namespace quakins {


template <typename idx_type,
          typename val_type, int fft_rank>
class FFT {

  cufftHandle plan_fwd, plan_bwd;
  val_type norm;

  using idx_array = std::array<idx_type,fft_rank>;

public:
  FFT(idx_array n, idx_type n_batch) {

    int nv[2] = {static_cast<int>(n[1]), static_cast<int>(n[0]-2)}; 
    int rnembed[2] = {static_cast<int>(n[1]), static_cast<int>(n[0])}; 
    int cnembed[2] = {static_cast<int>(n[1]), static_cast<int>(n[0]/2)}; 
 
    int rdist = static_cast<int>(n[0]*n[1]);
    int cdist = static_cast<int>(n[0]*n[1]/2);
    
    CUFFT_CALL(
    cufftPlanMany(&plan_fwd, fft_rank, nv, rnembed, 1, rdist,
                                           cnembed, 1, cdist,
                                           CUFFT_R2C, n_batch));
    CUFFT_CALL(
    cufftPlanMany(&plan_bwd, fft_rank, nv, cnembed, 1, cdist,
                                           rnembed, 1, rdist,
                                           CUFFT_C2R, n_batch));

    norm = static_cast<val_type>((n[0]-2)*n[1]);

  }

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
  void backward(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    auto real_ptr = reinterpret_cast<cufftReal*>(
                        thrust::raw_pointer_cast(&(*out_begin)));
    auto comp_ptr = reinterpret_cast<cufftComplex*>(
                        thrust::raw_pointer_cast(&(*in_begin)));
   
    CUFFT_CALL(cufftExecC2R(plan_bwd, comp_ptr, real_ptr));
  }


};
}
