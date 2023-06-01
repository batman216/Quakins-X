#pragma once

#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/adjacent_difference.h>
#include <fstream>


#include "crtp.hpp"

template <typename Iterator>
class strided_chunk_range
{
public:

  typedef typename thrust::iterator_difference<
                        Iterator>::type difference_type;
  struct stride_functor : 
  public thrust::unary_function<difference_type,difference_type> {
    difference_type stride;
    int chunk;
    stride_functor(difference_type stride, int chunk)
    : stride(stride), chunk(chunk) {}
    __host__ __device__
    difference_type operator()(const difference_type& i) const {
      int pos = i/chunk;
      return ((pos * stride) + (i-(pos*chunk)));
    }
  };

  typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
  typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  strided_chunk_range(Iterator first, Iterator last, 
                      difference_type stride, int chunk)
  : first(first), last(last), stride(stride), chunk(chunk) {assert(chunk<=stride);}

  iterator begin(void) const {
    return PermutationIterator(first,
              TransformIterator(CountingIterator(0),
                                stride_functor(stride, chunk)));
  }
  
  iterator end(void) const
  {
    int lmf = last-first;
    int nfs = lmf/stride;
    int rem = lmf-(nfs*stride);
    return begin() + (nfs*chunk) + ((rem<chunk)?rem:chunk);
  }

  protected:
    Iterator first;
    Iterator last;
    difference_type stride;
    int chunk;
};

template <typename idx_type, typename val_type, idx_type dim>
struct Parameters;

template <typename idx_type,
          typename val_type,
          idx_type dim,
          idx_type xdim, idx_type vdim>
class FreeStream {

  typedef thrust::tuple<val_type,val_type> TwoValTuple;
  typedef thrust::tuple<val_type,val_type,
                        val_type> ThreeValTuple;

  typedef thrust::tuple<val_type,val_type,
                        val_type,val_type> FourValTuple;

  const idx_type n_bd, nx, nv, n_tot;
  const val_type dx;

  // flux function and shift length of each grid
  thrust::host_vector<val_type> alpha;

public:
  FreeStream(Parameters<idx_type,val_type,dim> *p,val_type dt) :
  n_bd(p->n_ghost[xdim]), nx(p->n_tot_local[xdim]-p->n_ghost[xdim]*2), 
  nv(p->n_tot_local[vdim]), n_tot(p->n_1d_per_dev), dx(p->interval[xdim]) {
    
    val_type dv = p->interval[vdim];
    val_type dx = p->interval[xdim];
    val_type v_min = p->low_bound[vdim];

    alpha.resize(nv);

    auto v_itor_begin = 
         thrust::make_transform_iterator(
         thrust::make_counting_iterator(static_cast<idx_type>(0)),
         [v_min,dv,dx,dt](idx_type idx) {
           
           return dt*(v_min + static_cast<val_type>(dv*idx))/dx; 
         });

    thrust::copy(v_itor_begin,v_itor_begin+nv,alpha.begin());
    
  }

  template <typename itor_type>
  __host__
  void operator()(itor_type itor_begin, itor_type itor_end, 
                idx_type n_chunk,int gpu) {

    std::ofstream pout("phi"+std::to_string(gpu),std::ios::out);
    // the outermost dimension is calculated sequentially
    std::size_t n_step = n_tot/n_chunk; 

    // prepare the flux function
    thrust::device_vector<val_type> flux(n_chunk);
    
    // Boundary Condition --------------------------------------
    strided_chunk_range<itor_type> 
      left_inside(itor_begin+n_bd,itor_end-n_bd,nx+2*n_bd, n_bd);
    strided_chunk_range<itor_type> 
      left_outside(itor_begin,itor_end-n_bd,nx+2*n_bd, n_bd);
    strided_chunk_range<itor_type> 
      right_inside(itor_begin+nx,itor_end-n_bd,nx+2*n_bd, n_bd);
    strided_chunk_range<itor_type> 
      right_outside(itor_begin+nx+n_bd,itor_end-n_bd,nx+2*n_bd, n_bd);

    thrust::copy(thrust::device,
                 left_inside.begin(),left_inside.end(),
                 right_outside.begin());
    thrust::copy(thrust::device,
                 right_inside.begin(),right_inside.end(),
                 left_outside.begin());
    // ---------------------------------------------------------

    constexpr bool print_flux = 0;

    val_type a, shift;

    auto itor_begin_l = itor_begin;

    for (std::size_t i = 0; i<n_step/2; i++) {

      a = std::modf(alpha[i],&shift);
      
      shift = -shift;

      auto zitor_neg_begin 
            = make_zip_iterator(thrust::make_tuple(
                      itor_begin_l+shift,
                      itor_begin_l+shift+1,  
                      itor_begin_l+shift+2));
      
      //i calculate the flux function \flux
      thrust::transform(thrust::device, 
                        zitor_neg_begin+n_bd-1,
                        zitor_neg_begin-n_bd+n_chunk,
                        flux.begin()+n_bd-1,
      [a]__host__ __device__(ThreeValTuple tuple){
        return a*(thrust::get<1>(tuple) 
         -(1-a)*(1+a)/6.*(thrust::get<2>(tuple)-thrust::get<1>(tuple))
         -(2+a)*(1+a)/6.*(thrust::get<1>(tuple)-thrust::get<0>(tuple)));
      });
/*
      for (int k=1; k<=static_cast<int>(shift); ++k) {
        auto zitor_presum_begin
             = thrust::make_zip_iterator(thrust::make_tuple(
                              itor_begin_l+n_bd-1 + k,
                              flux.begin()+n_bd-1 ));
         std::cout << k << " ";

        thrust::for_each(thrust::device,
                       zitor_presum_begin, zitor_presum_begin+n_chunk+1,
                       []__host__ __device__ (TwoValTuple tuple) {
                          thrust::get<1>(tuple) += thrust::get<0>(tuple);  
                       });
      }
*/
      thrust::adjacent_difference(thrust::device,
                                  flux.begin(),flux.end(),
                                  flux.begin());

      if constexpr (print_flux) {
        thrust::copy(flux.begin(),flux.end(),
                   std::ostream_iterator<val_type>(pout," "));
        pout << std::endl;
      }

      using namespace thrust::placeholders;

      // calculate f[i](t+dt)=f[i](t) + flux[i-1/2] -flux[i+1/2]
      thrust::transform(flux.begin(),flux.end(),
                        itor_begin_l,itor_begin_l, _2 - _1);

      itor_begin_l += n_chunk;
    } // v < 0
      

    for (std::size_t i = n_step/2; i<n_step; i++) {
      
      a = std::modf(alpha[i],&shift);

      shift = -shift;

      auto zitor_pos_begin 
           = make_zip_iterator(thrust::make_tuple(
                       itor_begin_l+static_cast<int>(shift)-1,
                       itor_begin_l+static_cast<int>(shift),  
                       itor_begin_l+static_cast<int>(shift)+1));

      thrust::transform(thrust::device, 
                        zitor_pos_begin+n_bd-1,
                        zitor_pos_begin-n_bd+n_chunk,
                        flux.begin()+n_bd-1,
      [a]__host__ __device__(ThreeValTuple tuple){
        return a*(thrust::get<1>(tuple) 
         +(1-a)*(2-a)/6.*(thrust::get<2>(tuple)-thrust::get<1>(tuple))
         +(1-a)*(1+a)/6.*(thrust::get<1>(tuple)-thrust::get<0>(tuple)));
      });


      for (int k=-1; k>=static_cast<int>(shift); --k) {
        
        auto zitor_presum_begin
             = thrust::make_zip_iterator(thrust::make_tuple(
                              itor_begin_l+n_bd-1 + k,
                              flux.begin()+n_bd-1 ));

        std::cout << k << " ";
        thrust::for_each(thrust::device,
                       zitor_presum_begin, zitor_presum_begin+n_chunk+1,
                       []__host__ __device__ (TwoValTuple tuple) {
                          thrust::get<1>(tuple) += thrust::get<0>(tuple);  
                       });
      }

      thrust::adjacent_difference(thrust::device,
                                  flux.begin(),flux.end(),
                                  flux.begin());

      if constexpr (print_flux) {
        thrust::copy(flux.begin(),flux.end(),
                   std::ostream_iterator<val_type>(pout," "));
        pout << std::endl;
      }

      using namespace thrust::placeholders;
      // calculate f[i](t+dt)=f[i](t) + flux[i-1/2] -flux[i+1/2]
      thrust::transform(flux.begin(),flux.end(),
                        itor_begin_l,itor_begin_l, _2 - _1);

      itor_begin_l += n_chunk;
    } // v > 0
  }

};


