#pragma once 
#include <cufft.h>

#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                    \
    {                                                                         \
        auto status = static_cast<cufftResult>( call );                       \
        if ( status != CUFFT_SUCCESS )                                        \
            fprintf( stderr,                                                  \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed " \
                     "with "                                                  \
                     "code (%d).\n",                                          \
                     #call,                                                   \
                     __LINE__,                                                \
                     __FILE__,                                                \
                     status );                                                \
    }
#endif  // CUFFT_CALL

template <typename val_type>
struct FFT_traits;

template <>
struct FFT_traits<float> {
  const cufftType forward = CUFFT_R2C;
  const cufftType inverse = CUFFT_C2R;

};

template <>
struct FFT_traits<double> {
  const cufftType forward = CUFFT_D2Z;
  const cufftType inverse = CUFFT_Z2D;

};

template <>
struct FFT_traits<thrust::complex<float>> {
  const cufftType forward = CUFFT_C2C;
  const cufftType inverse = CUFFT_C2C;

};

template <>
struct FFT_traits<thrust::complex<double>> {
  const cufftType forward = CUFFT_Z2Z;
  const cufftType inverse = CUFFT_Z2Z;

};


template <typename val_type>
struct fft_pointer_traits;

template <>
struct fft_pointer_traits<float> {
  using first = cufftReal; 
  using second = cufftComplex; 
};

template <>
struct fft_pointer_traits<double> {
  using first = cufftDoubleReal; 
  using second = cufftDoubleComplex; 
};

template <>
struct fft_pointer_traits<thrust::complex<float>> {
  using first = cufftComplex; 
  using second = cufftComplex; 
};

template <>
struct fft_pointer_traits<thrust::complex<double>> {
  using first = cufftDoubleComplex; 
  using second = cufftDoubleComplex; 
};

