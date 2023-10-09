#pragma once

#include <thrust/device_vector.h>

#include <fstream>
#include <iostream>

#include <cufft.h>

#include "../ReorderCopy.hpp"
#include "../thrust_fancy.hpp"

namespace quakins {

namespace details {

template <typename idx_type,
          typename val_type,
          idx_type dim>
struct EfieldSolver {

  const std::array<idx_type,dim> n,n_bd;
  const std::array<val_type,dim> interval;
  idx_type ntot;
  
  EfieldSolver(std::array<idx_type,dim> n, 
               std::array<idx_type,dim> n_bd, 
               std::array<val_type,dim> interval) 
    : n(n), n_bd(n_bd), interval(interval){
      ntot = n[1]*n[0];
    }

  template <typename itor_type>
  void operator()(itor_type phi_begin, itor_type phi_end, itor_type phi_ext_begin,
                  std::array<itor_type,dim> E_begin) {
    this->operator()(phi_begin,phi_end,E_begin);

  }


  template <typename itor_type>
  void operator()(itor_type phi_begin, itor_type phi_end,
                  std::array<itor_type,dim> E_begin) {

    auto middle_begin = thrust::make_zip_iterator(thrust::make_tuple(
                                phi_begin+2-2,phi_begin+2-1,
                                phi_begin+2+1,phi_begin+2+2));
   

    
    using zipItor = thrust::tuple<val_type,val_type,val_type,val_type>;
    
    val_type C = 1./12./interval[0];
    thrust::transform(thrust::device, 
                      middle_begin, middle_begin+n[0]*n[1]-2, E_begin[0]+2,
                      [C]__host__ __device__(zipItor _tuple) {
                      return C*(-thrust::get<0>(_tuple) + 8.*thrust::get<1>(_tuple)
                                -8.*thrust::get<2>(_tuple) + thrust::get<3>(_tuple));
                      });

    val_type Cb = 0.5/interval[0];
  /*
    thrust::host_vector<val_type> _cof0_l(3);
    _cof0_l[0] = Cb*3.; _cof0_l[1] = -4.*Cb; _cof0_l[2] = Cb;
    thrust::host_vector<val_type> _cof0_r(3);
    _cof0_r[0] = -Cb; _cof0_r[1] = 4.*Cb; _cof0_r[2] = -3.*Cb;

    // r boundary
    thrust::device_vector<val_type> cof0_l = _cof0_l;
    thrust::device_vector<val_type> cof0_r = _cof0_r;
    for (int i=0; i<ntot; i+=n[0]) {
      
      E_begin[0][i]  = thrust::inner_product(thrust::device,
                                             cof0_l.begin(), cof0_l.end(), phi_begin+i,0);
      E_begin[0][i+1]= thrust::inner_product(thrust::device,
                                             cof0_l.begin(), cof0_l.end(), phi_begin+i+1,0);
    } // left
    for (int i=n[0]-2; i<ntot; i+=n[0]) {
      E_begin[0][i]  = thrust::inner_product(thrust::device,
                                             cof0_r.begin(), cof0_r.end(), phi_begin+i-2,0);
      E_begin[0][i+1]= thrust::inner_product(thrust::device,
                                             cof0_r.begin(), cof0_r.end(), phi_begin+i-1,0);
    } // right
*/
    int idx1,idx2,idx3,idx4;
    for (int i=0; i<n[1]; i++) {
      idx1 = i*n[0];
      idx2 = i*n[0]+1;
      idx3 = idx1+n[0]-2;
      idx4 = idx2+n[0]-2;

      E_begin[0][idx1] = -Cb*(-3.*phi_begin[idx1]+4.*phi_begin[idx1+1]-phi_begin[idx1+2]);
      E_begin[0][idx2] = -Cb*(-3.*phi_begin[idx2]+4.*phi_begin[idx2+1]-phi_begin[idx2+2]);
      E_begin[0][idx3] = -Cb*(phi_begin[idx3-2]-4.*phi_begin[idx3-1]+3.*phi_begin[idx3]);
      E_begin[0][idx4] = -Cb*(phi_begin[idx4-2]-4.*phi_begin[idx4-1]+3.*phi_begin[idx4]);
    }

    ReorderCopy<idx_type,val_type,dim> reorder1(n,{1,0}), reorder2({n[1],n[0]},{1,0});

    thrust::device_vector<val_type> phi_re(ntot),E1_buf(ntot);
    reorder1(phi_begin,phi_begin+ntot,phi_re.begin());
    
    auto middle_begin1 = thrust::make_zip_iterator(thrust::make_tuple(
                                phi_re.begin()+2-2,phi_re.begin()+2-1,
                                phi_re.begin()+2+1,phi_re.begin()+2+2));
 
    val_type C1 = 1./12./interval[1];
    thrust::transform(thrust::device, 
                      middle_begin1, middle_begin1+ntot-2, E1_buf.begin()+2,
                      [C1]__host__ __device__(zipItor _tuple) {
                      return C1*(-thrust::get<0>(_tuple) + 8.*thrust::get<1>(_tuple)
                                -8.*thrust::get<2>(_tuple) + thrust::get<3>(_tuple));
                      });

    val_type Cb1 = 0.5/interval[1];
    for (int i=0; i<n[0]; i++) {
      int idx1 = i*n[1];
      int idx2 = i*n[1]+1;
      int idx3 = idx1+n[1]-2;
      int idx4 = idx2+n[1]-2;

      E1_buf[idx1] = -Cb1*(-3.*phi_re[idx1]+4.*phi_re[idx1+1]-phi_re[idx1+2]);
      E1_buf[idx2] = -Cb1*(-3.*phi_re[idx2]+4.*phi_re[idx2+1]-phi_re[idx2+2]);
      E1_buf[idx3] = -Cb1*(phi_re[idx3-2]-4.*phi_re[idx3-1]+3.*phi_re[idx3]);
      E1_buf[idx4] = -Cb1*(phi_re[idx4-2]-4.*phi_re[idx4-1]+3.*phi_re[idx4]);
    }

    reorder2(E1_buf.begin(),E1_buf.end(),E_begin[1]);

  }

};

template <typename idx_type,
          typename val_type,
          idx_type dim>
class FourierSpectrumVeloSpace {

  cufftHandle plan_fwd, plan_bwd;

  const idx_type nx1,nx2, nx2loc, nv1, nv2, nx1bd, nx2bd;
  const val_type dv1, dv2, dx1, dx2;

  val_type dl1, dl2, dt;

  thrust::device_vector<val_type> r, Ex, Ey, lam1, lam2, phase;
  std::ofstream Eout;
  
  int gpu, gpu_size;

public:
  template <typename Parameters,typename ParallelCommunicator>
  FourierSpectrumVeloSpace(Parameters *p, 
                           ParallelCommunicator *para,
                           val_type dt,int xdim, int vdim) :
  nv1(p->n[0]), nv2(p->n[1]),
  nx1(p->n_all[2]), nx2(p->n_all[3]), nx2loc(p->n_all[3]/p->n_dev),
  dx1(p->interval[2]), dx2(p->interval[3]), dt(dt),
  dv1(p->interval[0]), dv2(p->interval[1]), 
  nx1bd(p->n_ghost[2]), nx2bd(p->n_ghost[3]) {

    dl1 = 2.*M_PI/(nv1-2)/p->interval[0]; 
    dl2 = 2.*M_PI/(nv2)/p->interval[1]; 
   
    int nx1nobd = nx1-2*nx1bd;
    int nx2nobd = nx2-12*nx2bd;
    Ex.resize(nx1nobd*nx2nobd);    
    Ey.resize(nx1nobd*nx2nobd);
    r.resize(nx1nobd*nx2nobd);

    lam1.resize(nv1*nv2/2); lam2.resize(nv1*nv2/2);
    phase.resize(nv1*nv2/2);
#ifdef CYLIND
    
    for (int j=0; j<nx2nobd; j++) {
      for (int i=0; i<nx1nobd; i++) {
        r[j*nx1nobd+i] = i*dx1 + .5*dx1;
      }
    }
#endif
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

    this->gpu = para->mpi_rank;
    this->gpu_size = para->mpi_size;
  }

  template <typename itor_type, typename vitor_type>
  void advance(itor_type itor_begin, itor_type itor_end, 
               vitor_type v_begin) {

    EfieldSolver<idx_type,val_type,2> 
      solveEfield({nx1-2*nx1bd,nx2-2*gpu_size*nx2bd},{2,2},{dx1,dx2});
    solveEfield(v_begin,v_begin+(nx1-2*nx1bd)*(nx2-2*gpu_size*nx2bd),v_begin,
                std::array<itor_type,2>{Ex.begin(),Ey.begin()});

#ifdef CYLIND
  
    thrust::transform(r.begin(),r.end(),Ex.begin(),Ex.begin(),
                      []__host__ __device__(val_type r,val_type F) 
                      {  return 0*F - 1.0/r; });
#endif

    auto comp_ptr = reinterpret_cast<cufftComplex*>(
                    thrust::raw_pointer_cast(&(*itor_begin)));

    val_type time_step = this->dt;
    val_type F1, F2;

    int jstart = gpu*nx2loc, i, I;
    for (int j1=nx2bd+jstart; j1<jstart+nx2loc-nx2bd; j1++) {
      for (int i1=nx1bd; i1<nx1-nx1bd; i1++) {

        i = i1 + j1*nx1 - gpu*nx1*nx2loc;
        I = i1-nx1bd + (j1-(2*gpu+1)*nx2bd)*(nx1-2*nx1bd);

        F1 = Ex[I]; F2 = Ey[I];
        
        thrust::transform(thrust::device,
                          lam1.begin(),lam1.end(),lam2.begin(),phase.begin(),
                          [time_step,F1,F2]__host__ __device__
                          (const val_type& l1, const val_type& l2) {
                            return (F1*l1+F2*l2)*time_step; 
                          });
        
        // f(t+dt) = f(t)exp(-i*phase)
        evolve_with_phase(comp_ptr + i*nv1*nv2/2,
                          comp_ptr + (i+1)*nv1*nv2/2,
                          phase.begin());                          

      }
    }   
    
  }
};


} // namespace details
  
} // namespace quakins
