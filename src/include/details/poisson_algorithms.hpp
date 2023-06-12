#pragma once

#include <cufft.h>


template<typename T>
struct cufftValTraits;

template<>
struct cufftValTraits<float> { using type = cufftReal; };

template<>
struct cufftValTraits<double> { using type = cufftDoubleReal; };


template<typename idx_type, typename val_type, idx_type dim>
class FFTandInv {

  cufftHandle plan_fwd, plan_inv;
  
  using cufftVal_t = typename cufftValTraits<val_type>::type;

  cufftVal_t *f1x, *f2x;
  cufftComplex *f1k, *f2k;
  idx_type size;

public:
  FFTandInv(std::array<idx_type,dim> n_dim,
            std::array<val_type,2*dim> bound) {
    
    size = 1;
    for (int i=0; i<dim; ++i) size *= n_dim[i];

    cufftPlanMany(&plan_fwd,
                  1,         // FFT dimension
                  (int*)&n_dim[1], // #data of each dimension 
                  NULL,      // #data of each dimension       -|
                             //    =n_dim if NULL              |
                  1,         // interval inside                |-input
                             //    ignored if previous is NULL | 
                  n_dim[1],  // interval between batches      _|
                  NULL,      // #data of each dimension       -|
                             //    =n_dim if NULL              |
                  n_dim[1],  // interval inside                |-output
                             //    ignored if previous is NULL |
                  1,         // interval between batches      _|
                  CUFFT_R2C, // cufft type
                  n_dim[0]   // #batch
                  );
  

    cufftPlanMany(&plan_inv,
                  1,         // FFT dimension
                  (int*)&n_dim[1], // #data of each dimension 
                  NULL,      // #data of each dimension       -|
                             //    =n_dim if NULL              |
                  n_dim[1],  // interval inside                |-input
                             //    ignored if previous is NULL | 
                  1,         // interval between batches      _|
                  NULL,      // #data of each dimension       -|
                             //    =n_dim if NULL              |
                  n_dim[1],  // interval inside                |-output
                             //    ignored if previous is NULL |
                  1,         // interval between batches      _|
                  CUFFT_C2R, // cufft type
                  n_dim[0]   // #batch
                  );
  

    cudaMalloc((void**)&f1k, sizeof(cufftComplex)*size);
    cudaMalloc((void**)&f2k, sizeof(cufftComplex)*size);
    
  }

  template <typename itor_type> __host__  
  void solve(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    f1x = (cufftVal_t*) thrust::raw_pointer_cast(&(*in_begin));  
    f2x = (cufftVal_t*) thrust::raw_pointer_cast(&(*out_begin));  


    cufftExecR2C(plan_fwd, f1x, f1k);
    
    thrust::copy(thrust::device, f1k, f1k+size, f2k);

    cufftExecC2R(plan_inv, f2k, f2x);

  }

};




template<typename idx_type, typename val_type, idx_type dim>
class FFT {
  

public:
  FFT(std::array<idx_type,dim> n_dim,
      std::array<val_type,2*dim> bound) {}

  template <typename itor_type> __host__  
  void solve(itor_type in_begin, itor_type in_end, itor_type out_begin) {

  }

};


