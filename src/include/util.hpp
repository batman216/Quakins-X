#pragma once 

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <string>
#include <map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// dynamic polyorphism
template <typename t_Product> 
class Factory {
public:
  virtual t_Product *produce() = 0;
protected:
  Factory() {}  
  virtual ~Factory() {}
};

/// forward declaration
template <typename t_Product> class Proxy; 
  
template <typename t_Product, typename t_ConcreteProduct> 
class ConcreteFactory : public Factory<t_Product> {
public:
  explicit ConcreteFactory(std::string p_name) {
    Proxy<t_Product>::Instance().registerProduct(this,p_name);
  }
  t_Product *produce() { return new t_ConcreteProduct; }

};

/**
 *   The class Proxy is like a flyweight factory
 */
template <typename t_Product>
class Proxy {
  Proxy() {}  /// private creator for singleton
  Proxy(const Proxy& other) {}  
public:
  /**
   *  The client fetch/purchase concrete product by p_name
   *  according to the regedit
   */
  std::map<std::string, Factory<t_Product>*> regedit;
  
  static Proxy<t_Product>& Instance() {
    static Proxy<t_Product> instance;
    /// Meyers Singleton: create an instance only when this function is called.
    return instance;
  }
  void registerProduct(Factory<t_Product>* reg, std::string p_name) {
    regedit[p_name] = std::move(reg);
  }

  t_Product* get(std::string p_name) { /// flyweight singleton
    if (regedit.find(p_name) != regedit.end())
      return regedit[p_name]->produce();
      /// produce  
    else {
      std::cout << "no product named " << p_name 
          << "registered." << std::endl;  
      return nullptr;
    }
  }
};


struct DoNotFlip {
  template <typename idx_type>
  __host__ __device__
  idx_type static cal(idx_type stride, std::size_t chunk, const idx_type& idx) {
    std::size_t pos = idx/chunk;
    return pos * stride + idx-(pos*chunk);
  }
};


struct Flip {
  template <typename idx_type>
  __host__ __device__
  idx_type static cal(idx_type stride, std::size_t chunk, const idx_type& idx) {
    std::size_t pos = idx/chunk;
    return pos * stride + chunk -idx+(pos*chunk)-1;
  }
};

template <typename Iterator, 
          typename Policy = DoNotFlip>
class strided_chunk_range
{
public:

  typedef typename thrust::iterator_difference<
                        Iterator>::type difference_type;
  struct stride_functor : 
  public thrust::unary_function<difference_type,difference_type> {
    difference_type stride;
    std::size_t chunk;

    stride_functor(difference_type stride, int chunk)
    : stride(stride), chunk(chunk) {}

    __host__ __device__
    difference_type operator()(const difference_type& i) const {

      // 0,1,..., chunk-1, stride+0, stride+1,..., stride+chunk-1, 
      // 2*stride+0, 2*stride+1,..., 2*stride+chunk-1, 

      return Policy::cal(stride,chunk,i);
    }
  };

  typedef typename thrust::counting_iterator
    <difference_type> CountingIterator;
  typedef typename thrust::transform_iterator
    <stride_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator
    <Iterator,TransformIterator> PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  strided_chunk_range(Iterator first, Iterator last, 
                      difference_type stride, int chunk)
  : first(first), last(last), stride(stride), chunk(chunk) { 
    assert(chunk<=stride); 
  }

  iterator begin(void) const {
    return PermutationIterator(first,
              TransformIterator(CountingIterator(0),
                                stride_functor(stride, chunk)));
  }
  
  iterator end(void) const
  {
    difference_type lmf = last-first;
    difference_type nfs = lmf/stride;
    difference_type rem = lmf-(nfs*stride);
    return begin() + (nfs*chunk) + ((rem<chunk)?rem:chunk);
  }

  protected:
    Iterator first;
    Iterator last;
    difference_type stride;
    int chunk;
};


// assign value from string
template <class A>
void assign(A& val, const std::string& name, 
    std::map<std::string, std::string> input) {
        
  std::stringstream buffer(input[name]);
  buffer >> val;
}

std::map<std::string, std::string> 
    read_box(std::ifstream& is,    
             std::string box_name);

template <typename T>
std::ostream& operator<<(std::ostream& os, const thrust::host_vector<T>& obj) {
  
  thrust::copy(obj.begin(), obj.end(),
            std::ostream_iterator<T>(os,"  "));
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const thrust::device_vector<T>& obj) {
  
  thrust::copy(obj.begin(), obj.end(),
            std::ostream_iterator<T>(os,"  "));
  return os;
}

// for one device per thread operation
uint64_t getHostHash(const char *);
void getHostName(char *, int);
