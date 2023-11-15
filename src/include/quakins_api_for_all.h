#pragma once
#include <thrust/complex.h>

namespace quakins {


template <typename complex_t, typename real_t>
real_t real(complex_t);

template <typename complex_t, typename imag_t>
imag_t imag(complex_t);

template <>
float real(float val) { return val; }

template <>
float imag(float val) { return 0; }


template <>
double real(double val) { return val; }

template <>
double imag(double val) { return 0; }

template <>
float real(thrust::complex<float> val) { return val.real(); }

template <>
float imag(thrust::complex<float> val) { return val.imag(); }

template <>
double real(thrust::complex<double> val) { return val.real(); }

template <>
double imag(thrust::complex<double> val) { return val.imag(); }


}

