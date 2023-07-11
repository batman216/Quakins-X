#pragma once 


namespace quakins {

template <typename idx_type,
          typename val_type, int fft_rank>
class FFT {

  cufftHandle plan_fwd, plan_bwd;
  val_type norm;

  using idx_array = std::array<idx_type,fft_rank>;

public:
  FFT(idx_array n, idx_type n_batch) {

    int nr[2] = {static_cast<int>(n[1]), static_cast<int>(n[0]-2)}; 
    int nv[2] = {static_cast<int>(n[1]), static_cast<int>(n[0]-2)}; 
    int nc[2] = {static_cast<int>(n[1]), static_cast<int>(n[0]/2)}; 
 
    int dist = static_cast<int>(n[0]*n[1]);
    int hdist = static_cast<int>(n[0]*n[1]/2);

    cufftPlanMany(&plan_fwd, fft_rank, nv, nr, 1, dist,
                                       nc, 1, hdist,
                                       CUFFT_R2C, 
                                       n_batch);
    cufftPlanMany(&plan_bwd, fft_rank, nv, nc, 1, hdist,
                                       nr, 1, dist,
                                       CUFFT_C2R, 
                                       n_batch);
    norm = static_cast<val_type>((n[0]-2)*n[1]);

  }
  template <typename itor_type>
  void forward(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    auto n_tot = in_end-in_begin;
    auto real_ptr = reinterpret_cast<cufftReal*>(
                        thrust::raw_pointer_cast(&(*in_begin)));
    auto comp_ptr = reinterpret_cast<cufftComplex*>(
                        thrust::raw_pointer_cast(&(*out_begin)));
 
    val_type Norm = norm;
    cufftExecR2C(plan_fwd, real_ptr, comp_ptr);
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
   
    cufftExecC2R(plan_bwd, comp_ptr, real_ptr);
  }


};
}
