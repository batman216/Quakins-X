#pragma once

#include <thrust/for_each.h>





template<typename idx_type,typename val_type,idx_type dim>
class TestParticle {

  thrust::device_vector<val_type> phi_p;

public:
  template<typename Parameters>
  TestParticle(Parameters *p, std::array<thrust::device_vector<val_type>,dim> x_coord) {
    int n1 = p->n[dim+0], n2 = p->n[dim+1];
    phi_p.resize(n1*n2);
    for (int i=0; i<p->n[dim+0];i++)
      for (int j=0; j<p->n[dim+1];j++) {
        
        phi_p[j*n1+i] = 0.02*erff(5.*x_coord[0][i])/
                           std::sqrt(std::pow(x_coord[0][i],2)
                                    +std::pow(x_coord[1][j]-30,2));

      }

  }

  template<typename itor_type>
  void operator()(itor_type it_begin,itor_type itor_end) {
    
    thrust::transform(thrust::device,
                     it_begin,itor_end,phi_p.begin(),it_begin,
                     []__host__ __device__(val_type phi_ind, val_type phi_ext)
                     {
                        return phi_ind + phi_ext;
                     });

  }

};
