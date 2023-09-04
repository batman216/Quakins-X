#pragma once

#include <cufft.h>
#include <cusolverSp.h>
#include <fftw3.h>
#include <Eigen/SparseLU>

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
            std::array<val_type,2*dim> bound,char coord) {
    
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
  fftw_plan plan_fwd, plan_inv;
  std::vector<double> f1x, f2x;
  std::vector<std::complex<double>> f1k, f2k;
  int size, fft_size, mat_size, nr, nz;
  
  // Eigen
  typedef Eigen::SparseMatrix<std::complex<double>> SparseMat;
  SparseMat A;
  Eigen::SparseLU<SparseMat, Eigen::COLAMDOrdering<int>> solver;
  
public:
  FFTandInvHost(std::array<idx_type,dim> n_dim,
                std::array<val_type,2*dim> bound, char coord) {

    nr = static_cast<int>(n_dim[0]);
    nz = static_cast<int>(n_dim[1]);
    val_type rmin = bound[0], zmin = bound[1];
    val_type rmax = bound[2], zmax = bound[3];

    val_type dk = 2.*M_PI/(zmax-zmin);
    val_type dr = (rmax-rmin)/nr;

    std::vector<val_type> r,k;
    r.reserve(nr); k.reserve(nz);

    for (int i=0;i<nr;i++)      r.push_back(rmin+i*dr+.5*dr);
    for (int i=0;i<=nz/2;i++)   k.push_back(i*dk);
    for (int i=1+nz/2;i<nz;i++) k.push_back((i-nz)*dk);

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
    int n[] = {nz};
    int howmany = nr; // number of transforms to compute

    fft_size = nz;
    size = nr*nz;
    
    f1x.resize(size);    f2x.resize(size);
    f1k.resize(size);    f2k.resize(size);

    // FFTW prepare
    plan_fwd = fftw_plan_many_dft_r2c(fft_rank, n, howmany,
                                      f1x.data(), n,
                                      1, nz, 
                                      reinterpret_cast<fftw_complex*>(f1k.data()), 
                                      n, nr,1,
                                      FFTW_MEASURE);

    plan_inv = fftw_plan_many_dft_c2r(fft_rank, n, howmany,
                                      reinterpret_cast<fftw_complex*>(f2k.data()), 
                                      n, nr,1,
                                      f2x.data(), n,
                                      nr,1,
                                      FFTW_MEASURE);
    // Eigen prepare
    mat_size = n_dim[0];
    A.resize(size,size);
    
    typedef typename Eigen::Triplet<val_type> Triplet;
    std::vector<Triplet> idxValList;
    idxValList.reserve(size*3-2*nz);

    val_type idr_d2 = coord=='d'?0:.5/dr, idr_s2 = 1./dr/dr;

    for (int I=0; I<nz; ++I) {
      // boundary condition at r=0
      if (coord=='d')
        idxValList.push_back(Triplet(I*nr,I*nr, 
                                  k[I]*k[I]+2*idr_s2+idr_d2/r[0]));
      else 
        idxValList.push_back(Triplet(I*nr,I*nr, 
                                  k[I]*k[I]+idr_s2+idr_d2/r[0]));

      for (int i=1; i<nr; ++i) {
        int s = i + I*nr;  
        idxValList.push_back(Triplet(s,s,
                                     2*idr_s2 + k[I]*k[I]));
        idxValList.push_back(Triplet(s-1,s,
                                     -idr_d2/r[i-1]-idr_s2));
        idxValList.push_back(Triplet(s,s-1,
                                     idr_d2/r[i]-idr_s2));
      }
      // zero b.c. at r=rmax
    }
    A.setFromTriplets(idxValList.begin(),idxValList.end());
    A.makeCompressed();
    solver.analyzePattern(A);
    solver.factorize(A);
    solver.compute(A);
  }

  template <typename itor_type>   
  void solve(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    std::copy(in_begin,in_end,f1x.begin());
    

    // substract ion
    std::for_each(f1x.begin(),f1x.end(),
                  [](auto& val) { val=1.0-val; });

    fftw_execute(plan_fwd);
    
    for (int i=0; i<size; ++i) { f1k[i] /= fft_size; }
    
    // create a map, do not copy
    Eigen::Map<Eigen::VectorXcd> b(f1k.data(),size),
                                 x(f2k.data(),size);

    x = solver.solve(b);
   
    fftw_execute(plan_inv);
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


