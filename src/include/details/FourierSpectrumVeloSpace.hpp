#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <fstream>

#include <cufft.h>

namespace quakins {

namespace details {


template <typename idx_type,
          typename val_type,
          idx_type dim,
          idx_type xdim, idx_type vdim>
class FourierSpectrumVeloSpace {

  cufftHandle plan_fwd, plan_bwd;


  const idx_type nx1,nx2, nx2loc, nv1, nv2;
  const val_type dv1, dv2;

  val_type dl1, dl2;

  thrust::device_vector<val_type> Ex, Ey, lam;

public:
  template <typename Parameters>
  FourierSpectrumVeloSpace(Parameters *p, val_type dt) :
  nv1(p->n[0]), nv2(p->n[1]),
  nx1(p->n_all[2]), nx2(p->n_all[3]), nx2loc(p->n_all[3]/p->n_dev),
  dv1(p->interval[0]), dv2(p->interval[1]) {

    dl1 = 2.*M_PI/p->length[0]; 
    dl2 = 2.*M_PI/p->length[1]; 
    
    Ex.resize(nx1*nx2);
    Ey.resize(nx1*nx2);
    lam.resize(nv1*nv2);

    for (int i=0; i<nv1; i++) {
      for (int j=0; j<nv2; j++) {
      }
    }
    
    int nv[2] = {static_cast<int>(nv1), static_cast<int>(nv2)}; 
    cufftPlanMany(&plan_fwd, 2, nv, NULL, 1, static_cast<int>(nv1*nv2),
                                    NULL, 1, static_cast<int>(nv1*nv2),
                                    CUFFT_R2C, static_cast<int>(nx1*nx2loc));
    cufftPlanMany(&plan_bwd, 2, nv, NULL, 1, static_cast<int>(nv1*nv2),
                                    NULL, 1, static_cast<int>(nv1*nv2),
                                    CUFFT_C2R, static_cast<int>(nx1*nx2loc));
  }

  template <typename itor_type, typename vitor_type>
  __host__ __device__
  void advance(itor_type itor_begin, itor_type itor_end, 
               vitor_type v_begin, int gpu) {
    
    auto main_pointer = thrust::raw_pointer_cast(&(*itor_begin));

    auto c_pointer = reinterpret_cast<cufftComplex*>(main_pointer);
    
    cufftExecR2C(plan_fwd, (cufftReal *) main_pointer,
                           (cufftComplex *) main_pointer);
    
    idx_type norm = nv1*nv2, n_tot = nv1*nv2*nx1*nx2loc;
    
    thrust::for_each(thrust::device,
                     c_pointer,c_pointer+n_tot,[norm]
                     __host__ __device__ (cufftComplex& val)
                     { val.x/=norm; val.y/=norm; });


    cufftExecC2R(plan_bwd, (cufftComplex *) main_pointer,
                           (cufftReal *) main_pointer);

  }

};


} // namespace details
} // namespace quakins
