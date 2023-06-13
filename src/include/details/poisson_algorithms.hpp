#pragma once

#include <cufft.h>
#include <cusolverSp.h>
#include <fftw3.h>
#include <Eigen/SparseCholesky>

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
  

  cusolverSpHandle_t plan_sol;
  int csrRowA, *csrColA;
  cuComplex csrValA;

public:
  FFTandInv(std::array<idx_type,dim> n_dim,
            std::array<val_type,2*dim> bound) {
    
    cufftCreate(&plan_fwd);
    cufftCreate(&plan_inv);

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
  

    cusolverSpCreate(&plan_sol);
    
  }

  template <typename itor_type> __host__  
  void solve(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    f1x = (cufftVal_t*) thrust::raw_pointer_cast(&(*in_begin));  
    f2x = (cufftVal_t*) thrust::raw_pointer_cast(&(*out_begin));  


    cufftExecR2C(plan_fwd, f1x, f1k);
    
    /*
     *
     * find a cuSolver to fill 
     *
     */
    


    cufftExecC2R(plan_inv, f2k, f2x);

  }

};





template<typename idx_type, typename val_type, idx_type dim>
class FFTandInvHost {

  // FFTW
  fftwf_plan plan_fwd, plan_inv;
  std::vector<val_type> f1x, f2x, k;
  std::vector<std::complex<val_type>> f1k, f2k;
  int size, fft_size, mat_size;
  
  // Eigen
  Eigen::SparseMatrix<val_type> A;
  Eigen::VectorXf b, x;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<val_type>> solver;
  
public:
  FFTandInvHost(std::array<idx_type,dim> n_dim,
                std::array<val_type,2*dim> bound) {

    val_type dk = 2.*M_PI/(bound[3]-bound[1]);

   /**************************************************************************
    * parameters for advanced FFTW in/output (similar to cufftmany)
    * howmany is the (nonnegative) number of transforms to compute. 
    * The resulting plan computes howmany transforms, where the input of 
    * the k-th transform is at location in+k*idisti (in C pointer arithmetic), 
    * and its output is at location out+k*odist. Plans obtained in
    * this way can often be faster than calling FFTW multiple times 
    * for the individual transforms. The basic fftw_plan_dft interface 
    * corresponds to howmany=1 (in which case the dist parameters are ignored).
    ***************************************************************************/
    
    int fft_rank = 1;
    int n[] = {n_dim[1]};
    int howmany = n_dim[0]; // number of transforms to compute

    fft_size = n_dim[1];
    size = 1;
    for (int i=0; i<dim; ++i) size *= n_dim[i];

    
    f1x.resize(size);    f2x.resize(size);
    f1k.resize(size);    f2k.resize(size);

    // FFTW prepare
    plan_fwd = fftwf_plan_many_dft_r2c(fft_rank, n, howmany,
                                      f1x.data(), n,
                                      1, n_dim[1], 
                                      reinterpret_cast<fftwf_complex*>(f1k.data()), n,
                                      n_dim[0],1,
                                      FFTW_MEASURE);

    plan_inv = fftwf_plan_many_dft_c2r(fft_rank, n, howmany,
                                      reinterpret_cast<fftwf_complex*>(f2k.data()), n,
                                      n_dim[0],1,
                                      f2x.data(), n,
                                      n_dim[0],1,
                                      FFTW_MEASURE);
    // Eigen prepare
    mat_size = n_dim[0];
    A.resize(size,size);
    b.resize(size);
    
    typedef typename Eigen::Triplet<val_type> Triplet;
    std::vector<Triplet> idxValList;
    idxValList.reserve(size);

    for (int i=0; i<size; ++i)
      idxValList.push_back(Triplet(i,i,1.));
    A.setFromTriplets(idxValList.begin(),idxValList.end());

  }

  template <typename itor_type>   
  void solve(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    std::copy(in_begin,in_end,f1x.begin());
    fftwf_execute(plan_fwd);
    
    solver.compute(A);
    Eigen::Map<Eigen::VectorXcf> b(f1k.data(),size),
                                 x(f2k.data(),size);
   
    x = solver.solve(b);

    fftwf_execute(plan_inv);
    std::copy(f2x.begin(),f2x.end(),out_begin);
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


