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
    
    std::puts("Poisson solver is cylindrical.");

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
  std::vector<val_type> r,k,rr;
  int size, fft_size, mat_size, nr, nz;
  
  // Eigen
  typedef Eigen::SparseMatrix<std::complex<double>> SparseMat;
  SparseMat A;
  Eigen::SparseLU<SparseMat, Eigen::COLAMDOrdering<int>> solver;
  
public:
  FFTandInvHost(std::array<idx_type,dim> n_dim,
                std::array<val_type,2*dim> bound) {

    nr = static_cast<int>(n_dim[0]);
    nz = static_cast<int>(n_dim[1]);
    val_type rmin = bound[0], zmin = bound[1];
    val_type rmax = bound[2], zmax = bound[3];

    val_type dk = 2.*M_PI/(zmax-zmin);
    val_type dr = (rmax-rmin)/nr;

    r.reserve(nr); k.reserve(nz);
    rr.reserve(nr*nz);
    
    for (int i=0;i<nr*nz;i++)   
      r.push_back(rmin+i*dr+.5*dr);

    for (int j=0;j<nz;j++) for (int i=0;i<nr;i++)  
      rr.push_back(r[j+i*nz]);
    
    for (int i=0;i<=nz/2;i++)   
      k.push_back(i*dk);
    for (int i=1+nz/2;i<nz;i++) 
      k.push_back((i-nz)*dk);

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

    /*****************************************************************
    fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany,
                                 fftw_complex *in, const int *inembed,
                                 int istride, int idist,
                                 fftw_complex *out, const int *onembed,
                                 int ostride, int odist,
                                 int sign, unsigned flags);
    ******************************************************************/
    // FFTW prepare
    plan_fwd = fftw_plan_many_dft_r2c(fft_rank, n, howmany,
                                      f1x.data(), n, 1, nz, 
                                      reinterpret_cast<fftw_complex*>(f1k.data()), 
                                      n, nr, 1,
                                      FFTW_MEASURE);

    plan_inv = fftw_plan_many_dft_c2r(fft_rank, n, howmany,
                                      reinterpret_cast<fftw_complex*>(f2k.data()), 
                                      n, nr, 1,
                                      f2x.data(), n, nr, 1,
                                      FFTW_MEASURE);
    // Eigen prepare
    mat_size = n_dim[0];
    A.resize(size,size);
    
    typedef typename Eigen::Triplet<val_type> Triplet;
    std::vector<Triplet> idxValList;
    idxValList.reserve(size*3-2*nz);

#ifdef HIGH_POISS
    std::puts("Poisson solver high accuracy.");
    val_type is2 = 1.0/(dr*dr*12.0);
    for (int I=0; I<nz; ++I) {
      // boundary condition at r=0
      idxValList.push_back(Triplet(I*nr,I*nr, 
                                k[I]*k[I] + 30.0*is2));
      idxValList.push_back(Triplet(I*nr,I*nr+1, 
                                (-15.0-9.0*dr/r[0])*is2));
      idxValList.push_back(Triplet(I*nr+1,I*nr, 
                                (-15.0+7.0*dr/r[1])*is2));
      idxValList.push_back(Triplet(I*nr+1,I*nr+1, 
                                k[I]*k[I]+ 30.*is2));

      for (int i=2; i<nr; ++i) {
        int s = i + I*nr;  
        if (i==nr-1)
          idxValList.push_back(Triplet(s,s,
                                     (31.0+dr/r[i])*is2+ k[I]*k[I]));
        else
          idxValList.push_back(Triplet(s,s,
                                     30.*is2 + k[I]*k[I]));

        idxValList.push_back(Triplet(s-1,s,
                                     (-16.0-8.0*dr/r[i-1])*is2));
        idxValList.push_back(Triplet(s,s-1,
                                     (-16.0+8.0*dr/r[ i ])*is2));
        idxValList.push_back(Triplet(s-2,s,
                                     (  1.0+1.0*dr/r[i-2])*is2));
        idxValList.push_back(Triplet(s,s-2,
                                     (  1.0-1.0*dr/r[ i ])*is2));

      }

      // zero b.c. at r=rmax

    }

#else
    val_type idr_d2 = .5/dr, idr_s2 = 1./dr/dr;
    for (int I=0; I<nz; ++I) {
      // boundary condition at r=0
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
#endif
    A.setFromTriplets(idxValList.begin(),idxValList.end());
    A.makeCompressed();
    solver.analyzePattern(A);
    solver.factorize(A);
    solver.compute(A);
  }

  template <typename itor_type>   
  void solve(itor_type in_begin, itor_type in_end, itor_type out_begin) {

    // substract ion
    std::transform(in_begin,in_end,rr.begin(),f1x.begin(),
                  [](auto val, val_type r) { return (val); });

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
class FFT1D {
  
  // FFTW
  fftw_plan plan_fwd, plan_inv;
  int fft_size, real_size,comp_size, nx, nxh;

  std::complex<double> *pk;
  double *px;
 
  std::vector<double> inverse_k_square;

public:
  FFT1D(std::array<idx_type,dim> n_dim,
      std::array<val_type,2*dim> bound) {
 
    nx = static_cast<int>(n_dim[0]);
    nxh = nx/2+1;

    val_type xmin = bound[0];
    val_type xmax = bound[1];

    val_type dkx = 2.*M_PI/(xmax-xmin);

    inverse_k_square.resize(nxh);

    inverse_k_square[0] = 0;

    val_type kx_sq;
    for (int j=1; j<nx; j++) {
      kx_sq = j<=nx/2? std::pow(j*dkx,2)
                     : std::pow((j-nx)*dkx,2);
      inverse_k_square[j] = 1/kx_sq/kx_sq;
    }

    fft_size  = nx;
    real_size = nx;     
    comp_size = nxh;  

    pk = new std::complex<double>[comp_size];
    px = new double[real_size];

    plan_fwd = fftw_plan_dft_r2c_1d(nx,px,reinterpret_cast<fftw_complex*>(pk),FFTW_MEASURE);
    plan_inv = fftw_plan_dft_c2r_1d(nx,reinterpret_cast<fftw_complex*>(pk),px,FFTW_MEASURE);

  }

  template <typename itor_type> 
  void solve(itor_type in_begin, itor_type in_end, itor_type out_begin) {
 
    std::copy(in_begin,in_end,px);

    // substract ion
    std::for_each(px,px+real_size, [](auto& val) { val-=1.0; });

    fftw_execute(plan_fwd);

    for (int i=0; i<comp_size; ++i) { pk[i] *= -inverse_k_square[i]/fft_size;  }
    
    fftw_execute(plan_inv);
    std::copy(px,px+real_size,out_begin);
      
  }

};

template<typename idx_type, typename val_type, idx_type dim>
class FFT2D_Cart {
  
  // FFTW
  fftw_plan plan_fwd, plan_inv;
  int fft_size, real_size,comp_size, nx, nxh, ny, nyh;

  std::complex<double> *pk;
  double *px;
 
  std::vector<double> inverse_k_square;

public:
  FFT2D_Cart(std::array<idx_type,dim> n_dim,
      std::array<val_type,2*dim> bound) {
 
    nx = static_cast<int>(n_dim[0]);
    nxh = nx/2+1;
    ny = static_cast<int>(n_dim[1]);
    nyh = ny/2+1;

    val_type xmin = bound[0], ymin = bound[1];
    val_type xmax = bound[2], ymax = bound[3];

    val_type dkx = 2.*M_PI/(xmax-xmin);
    val_type dky = 2.*M_PI/(ymax-ymin);

    inverse_k_square.resize(nxh*ny);

    val_type kx_sq, ky_sq;

    inverse_k_square[0] = 0;
    for (int i=1; i<nyh; i++) {
      ky_sq = std::pow(i*dky,2);
      inverse_k_square[i] = 1./ky_sq;
    }

    for (int j=1; j<nx; j++) {
      kx_sq = j<=nx/2? std::pow(j*dkx,2)
                     : std::pow((j-nx)*dkx,2);
      for (int i=0; i<nyh; i++) {
        ky_sq = std::pow(i*dky,2);
        inverse_k_square[j*nyh+i] = 1./(kx_sq+ky_sq);
      }
    }

    fft_size  = nx*ny;
    real_size = nx*ny;     
    comp_size = nx*nyh;  

    pk = new std::complex<double>[comp_size];
    px = new double[real_size];

    plan_fwd = fftw_plan_dft_r2c_2d(nx,ny,px,reinterpret_cast<fftw_complex*>(pk),FFTW_MEASURE);
    plan_inv = fftw_plan_dft_c2r_2d(nx,ny,reinterpret_cast<fftw_complex*>(pk),px,FFTW_MEASURE);

  }

  template <typename itor_type> 
  void solve(itor_type in_begin, itor_type in_end, itor_type out_begin) {
 
    std::copy(in_begin,in_end,px);

    // substract ion
    std::for_each(px,px+real_size, [](auto& val) { val-=1.0; });

    fftw_execute(plan_fwd);
/*      
    std::ofstream ko("ktest.qout",std::ios::out);
    for (int i=0; i<comp_size; ++i)  
      ko << inverse_k_square[i] << "\t" << pk[i].real() << "\t" << pk[i].imag() << std::endl;
    ko.close();
 */     
    for (int i=0; i<comp_size; ++i) { pk[i] *= -inverse_k_square[i]/fft_size;  }
    
    fftw_execute(plan_inv);
    std::copy(px,px+real_size,out_begin);
      
  }

};



