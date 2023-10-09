#pragma once

#include <thrust/for_each.h>





template<typename idx_type,typename val_type,idx_type dim>
class TestParticle {

  thrust::device_vector<val_type> phi_p;
  const int n1, n2;
  std::array<thrust::device_vector<val_type>,dim> x_coord;
public:
  template<typename Parameters>
  TestParticle(Parameters *p, std::array<thrust::device_vector<val_type>,dim> x_coord) :
  x_coord(x_coord), n1(p->n[dim+0]), n2(p->n[dim+1]) {
    phi_p.resize(n1*n2);
  }

  template<typename itor_type>
  void operator()(itor_type it_begin,itor_type itor_end,val_type t) {
    
    for (int i=0; i<n1;i++)
      for (int j=0; j<n2;j++) {
        
        phi_p[j*n1+i] = 2.0*erff(12.*x_coord[0][i])/
                           sqrtf(powf(x_coord[0][i],2)
                                    +powf(x_coord[1][j]-1.0*t,2));

      }


    thrust::transform(thrust::device,
                     it_begin,itor_end,phi_p.begin(),it_begin,
                     []__host__ __device__(val_type phi_ind, val_type phi_ext)
                     {
                        return phi_ind + phi_ext;
                     });

  }

};
