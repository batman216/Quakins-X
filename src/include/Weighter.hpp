#pragma once


namespace quakins {


template<typename idx_type, typename val_type,idx_type dim>
struct Weighter {

  std::array<thrust::device_vector<val_type>,dim> v;
  thrust::device_vector<val_type> v_square;
  std::array<idx_type,dim> nv,nx;
  std::array<val_type,dim> dv,dx,vmin;
  idx_type v_tot, x_tot, n_tot;
  int rank;

  template <typename Parameters>
  __host__ 
  Weighter(Parameters *p,int rank) : rank(rank) {
    for (int i=0; i<dim;i++) {
      nv[i] = p->n[i];
      nx[i] = p->n_all_local[i+dim];
      dv[i] = p->interval[i];
      dx[i] = p->interval[i+dim];
      vmin[i] = p->low_bound[i];
    }
    v_tot = 1; x_tot = 1; 
    for (int i=0; i<dim;i++) {
      v_tot *= nv[i];
      x_tot *= nx[i];
    }
    n_tot = v_tot*x_tot;

    for (int i=0; i<dim;i++) 
      v[i].resize(v_tot); 
    
  }

  template <typename itor_type>
  void vSquare(itor_type in_begin, itor_type in_end, itor_type out_begin) {
   
    int nin1 = nv[0], nin2 = nv[1];
    val_type dv1 = dv[0], dv2 = dv[1],lb1 = vmin[0], lb2 = vmin[1];

    thrust::transform(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(static_cast<int>(v_tot)),
                      v[0].begin(),
                      [nin1,dv1,lb1]__host__ __device__(int idx) { 
                        return static_cast<val_type>(idx%nin1)*dv1 + lb1;
                      });

    thrust::transform(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(static_cast<int>(v_tot)),
                      v[1].begin(),
                      [nin2,dv2,lb2]__host__ __device__(int idx) { 
                        return static_cast<val_type>(idx%nin2)*dv2 + lb2;
                      });

    v_square.resize(v_tot);
    thrust::transform(thrust::device,
                      v[0].begin(),v[0].end(),v[1].begin(), v_square.begin(),
                      []__host__ __device__(val_type v1, val_type v2) { 
                        return powf(v1,2)+powf(v2,2);
                      });
  
    idx_type __vtot = v_tot;
    auto titor_begin = thrust::make_permutation_iterator(v_square.begin(),
                        thrust::make_transform_iterator(
                          thrust::make_counting_iterator(static_cast<idx_type>(0)),
                          [__vtot]__host__ __device__(idx_type idx){ return idx%__vtot; }));


    thrust::transform(in_begin, in_end, 
                      titor_begin,
                      out_begin,
                      []__host__ __device__(val_type val1, val_type val2) {
                        return val1*val2;
                      });
    
  }


};

}
