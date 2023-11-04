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
class FFT;


template <typename idx_type, typename val_type>
class FFT<idx_type,val_type,1> {

  cufftHandle plan_fwd, plan_bwd;
  val_type norm;

  idx_type n, n_batch; int  nr, nc;
public:
  FFT(idx_type n, idx_type n_batch) : n(n), n_batch(n_batch), 
  nr(static_cast<int>(n)), nc(static_cast<int>(n/2+1)) {
   
    int Nfft[1] = {nr};

    int Nrnem[1] = {nr}, Ncnem[1] = {nc};
    
    CUFFT_CALL(
    cufftPlanMany(&plan_fwd, 1, Nfft, Nrnem, 1, nr,
                                      Ncnem, 1, nc,
                                      CUFFT_R2C, n_batch));
    CUFFT_CALL(
    cufftPlanMany(&plan_bwd, 1, Nfft, Ncnem, 1, nc,
                                      Nrnem, 1, nr,
                                      CUFFT_C2R, n_batch));
    norm = static_cast<val_type>(n);

  }

  ~FFT() {
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_bwd);
  }

  template <typename itor_type>
  void forward(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    auto real_ptr = reinterpret_cast<cufftReal*>(
                        thrust::raw_pointer_cast(&(*in_begin)));
    auto comp_ptr = reinterpret_cast<cufftComplex*>(
                        thrust::raw_pointer_cast(&(*out_begin)));
 
    val_type Norm = norm;
    CUFFT_CALL(cufftExecR2C(plan_fwd, real_ptr, comp_ptr));

    auto c_begin = thrust::device_pointer_cast(comp_ptr);

    thrust::for_each(c_begin, c_begin + nc*n_batch,
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


template <typename idx_type, typename val_type>
class FFT<idx_type,val_type,2> {

  cufftHandle plan_fwd, plan_bwd;
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
      cufftPlanMany(&plan_bwd, 2, nv, cnembed, 1, cdist,
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
  void backward(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    auto real_ptr = reinterpret_cast<cufftReal*>(
                        thrust::raw_pointer_cast(&(*out_begin)));
    auto comp_ptr = reinterpret_cast<cufftComplex*>(
                        thrust::raw_pointer_cast(&(*in_begin)));
   
    CUFFT_CALL(cufftExecC2R(plan_bwd, comp_ptr, real_ptr));

  }

};


}
