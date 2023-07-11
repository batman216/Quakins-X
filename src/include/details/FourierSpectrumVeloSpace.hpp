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

  val_type dl1, dl2, dt;

  thrust::device_vector<val_type> Ex, Ey, lam1,lam2;

public:
  template <typename Parameters>
  FourierSpectrumVeloSpace(Parameters *p, val_type dt) :
  nv1(p->n[0]), nv2(p->n[1]),
  nx1(p->n_all[2]), nx2(p->n_all[3]), nx2loc(p->n_all[3]/p->n_dev),
  dv1(p->interval[0]), dv2(p->interval[1]), dt(dt) {

    dl1 = 2.*M_PI/(nv1-2)/p->interval[0]; 
    dl2 = 2.*M_PI/(nv2)/p->interval[1]; 
    
    Ex.resize(nx1*nx2);     Ey.resize(nx1*nx2);
    lam1.resize(nv1*nv2/2); lam2.resize(nv1*nv2/2);

    for (int j=0; j<nv2/2; j++) {
      for (int i=0; i<nv1/2; i++) {
        lam1[j*nv1/2+i] = i*dl1;
        lam2[j*nv1/2+i] = j*dl2;
      }
    }
    for (int j=nv2/2; j<nv2; j++) {
      for (int i=0; i<nv1/2; i++) {
        lam1[j*nv1/2+i] = i*dl1;
        lam2[j*nv1/2+i] = (j-static_cast<int>(nv2))*dl2;
      }
    }
    std::ofstream tout("lambda",std::ios::out);
    thrust::copy(lam1.begin(),lam1.end(),std::ostream_iterator<val_type>(tout," "));
    tout << std::endl;
    thrust::copy(lam2.begin(),lam2.end(),std::ostream_iterator<val_type>(tout," "));
    tout << std::endl;
  }
  template <typename itor_type, typename vitor_type>
  __host__ __device__
  void advance(itor_type itor_begin, itor_type itor_end, 
               vitor_type v_begin, int gpu) {

    auto comp_ptr = reinterpret_cast<cufftComplex*>(
                    thrust::raw_pointer_cast(&(*itor_begin)));

    val_type time_step = this->dt;
    for (int i=0; i<nx1*nx2loc; i++) {
   
      thrust::transform(thrust::device,
                        comp_ptr + i*nv1*nv2/2,
                        comp_ptr + (i+1)*nv1*nv2/2,
                        lam1.begin(),
                        comp_ptr + i*nv1*nv2/2,
                        [time_step]__host__ __device__
                        (cufftComplex val, const val_type& lam) {
                          cufftComplex next_val;
                          val_type phase = -1*time_step*lam;
                          next_val.x =  val.x*cos(phase) - val.y*sin(phase);
                          next_val.y =  val.x*sin(phase) + val.y*cos(phase);
                          return next_val;
                        });

    }
   
  }

};


} // namespace details
  
} // namespace quakins
