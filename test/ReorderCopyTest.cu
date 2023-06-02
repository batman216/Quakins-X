#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <cstdio>
#include <string>

#include "../src/include/Timer.hpp"
#include "../src/include/ReorderCopy.hpp"

using Real = float;

const std::size_t dim = 4;


int main(int argc, char* argv[]) {
  
  std::size_t n = std::stoi(argv[1]);

  std::size_t N = std::pow(n,dim);

  Timer watch;

  thrust::host_vector<Real> vec(N);
  srand(13);

  thrust::generate(vec.begin(),vec.end(),rand);
  thrust::device_vector<Real> vec1 = vec, vec2(N);


  std::array<std::size_t,4> order1 = {1,0,3,2};
  std::array<std::size_t,4> n_dim = {4*n,n,n/4,n};

  quakins::ReorderCopy<std::size_t,Real,dim> copy1(n_dim,order1);
  
  watch.tick("start... N="+std::to_string(N));
  copy1(vec1.begin(),vec1.end(),vec2.begin());
  watch.tock();


}
