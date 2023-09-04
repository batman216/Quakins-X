#pragma once

#include "../thrust_fancy.hpp"
#include <cuda_runtime.h>
#include <texture_types.h>

#include <cuda_texture_types.h>


namespace quakins {

namespace details {

template <typename idx_type,
          typename val_type,
          idx_type dim,
          idx_type xdim, idx_type vdim>
class WignerTerm {

  cufftHandle plan_fwd, plan_bwd;

  const idx_type nx1,nx2, nx2loc, nv1, nv2, nx1bd, nx2bd;
  const val_type dv1, dv2, dx1, dx2;

  val_type dl1, dl2, dt, L1,L2;

  thrust::device_vector<val_type> Ex, Ey, lam1,lam2, phase;
  std::ofstream Eout;

public:
  template <typename Parameters>
  WignerTerm(Parameters *p, val_type dt) :
  nv1(p->n[0]), nv2(p->n[1]),L1(p->length[2]),L2(p->length[3]),
  nx1(p->n_all[2]), nx2(p->n_all[3]), nx2loc(p->n_all[3]/p->n_dev),
  dx1(p->interval[2]), dx2(p->interval[3]), dt(dt),
  dv1(p->interval[0]), dv2(p->interval[1]), 
  nx1bd(p->n_ghost[2]), nx2bd(p->n_ghost[3]) {

    Eout.open("testE.qout",std::ios::out);
    dl1 = 2.*M_PI/(nv1-2)/p->interval[0]; 
    dl2 = 2.*M_PI/(nv2)/p->interval[1]; 
    
    Ex.resize((nx1-2*nx1bd)*(nx2-12*nx2bd));    
    Ey.resize((nx1-2*nx1bd)*(nx2-12*nx2bd));

    lam1.resize(nv1*nv2/2); lam2.resize(nv1*nv2/2);
    phase.resize(nv1*nv2/2); 

    for (int j=0; j<=nv2/2; j++) {
      for (int i=0; i<nv1/2; i++) {
        lam1[j*nv1/2+i] = i*dl1;
        lam2[j*nv1/2+i] = j*dl2;
      }
    }

    for (int j=nv2/2+1; j<nv2; j++) {
      for (int i=0; i<nv1/2; i++) {
        lam1[j*nv1/2+i] = i*dl1;
        lam2[j*nv1/2+i] = (j-static_cast<int>(nv2))*dl2;
      }
    }

  }

  template <typename itor_type, typename vitor_type>
  void advance(itor_type itor_begin, itor_type itor_end, 
               vitor_type v_begin, int gpu) {
    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<val_type>();
    // cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t phi_as_texture;
    cudaMallocArray(&phi_as_texture, &channelDesc, nx1, nx2);

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = nx1 * sizeof(val_type);
    // Copy data located at address h_data in host memory to device memory
    cudaMemcpy2DToArray(phi_as_texture, 0, 0, thrust::raw_pointer_cast(&(*v_begin)), 
                        spitch, nx1 * sizeof(val_type),
                        nx2, cudaMemcpyDeviceToDevice);
    // Specify texture
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = phi_as_texture;

    // Specify texture object parameters
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);


    auto comp_ptr = reinterpret_cast<cufftComplex*>(
                    thrust::raw_pointer_cast(&(*itor_begin)));

    val_type time_step = this->dt;

    int jstart = gpu*nx2loc, i, I;
    val_type X1, X2;
    for (int j1=nx2bd+jstart; j1<jstart+nx2loc-nx2bd; j1++) {
      for (int i1=nx1bd; i1<nx1-nx1bd; i1++) {
        
        X1 = i1 * dx1; X2 = j1 * dx2; 

        i = i1 + j1*nx1 - gpu*nx1*nx2loc;
        I = i1-nx1bd + (j1-(2*gpu+1)*nx2bd)*(nx1-2*nx1bd);
  
        auto l_begin = thrust::make_zip_iterator(
                      thrust::make_tuple(lam1.begin(),lam2.begin()));
        
        using Tuple = thrust::tuple<val_type,val_type>;

        val_type Q = 0.1, LL1 = L1, LL2 = L2;
        
        thrust::transform(thrust::device,
                          lam1.begin(), lam1.end(), lam2.begin(), phase.begin(),
                          [time_step,Q,X1,X2,LL1,LL2,texObj] __host__ __device__
                          (const val_type ll1, const val_type ll2) {
                            return -time_step*(tex2D<val_type>(texObj, (X1+ll1)/LL1,(X2+ll2)/LL2)
                                             -tex2D<val_type>(texObj, (X1-ll1)/LL1,(X2-ll2)/LL2));
                          });

        evolve_with_phase(comp_ptr + i*nv1*nv2/2,
                          comp_ptr + (i+1)*nv1*nv2/2,
                          phase.begin());

      }
    }   
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(phi_as_texture);
  }
};



} // details

} // quakins
