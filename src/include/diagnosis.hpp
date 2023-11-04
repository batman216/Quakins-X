#pragma once

#include "Integrator.hpp"
#include <fstream>
#include <iostream>
#include <thrust/copy.h>

namespace quakins {

namespace diagnosis {


template <typename T,typename val_type>
std::ostream& operator<<(std::ostream& os, const T& obj) {

  thrust::copy(obj.begin(),obj.end(),
               std::ostream_iterator<val_type>(os,"\t"));
  return os;

}

  
} // namespace diagnosis


template <typename idx_type, typename val_type, int dim_x, int dim_v>
struct Parameters;

template <typename idx_type, typename val_type, int dim_x, int dim_v>
class Probe {

  using Para = Parameters<idx_type,val_type,dim_x,dim_v>;

  Para *p;
  const int mpi_size, mpi_rank;
  std::ofstream fcout, p_os, e_os, d_os;

public:
  Probe(Para *p) : p(p),mpi_size(p->mpi_size),mpi_rank(p->mpi_rank) {  
    
    fcout.open("ff@"+std::to_string(mpi_rank)+".qout"); 
    p_os.open("potn"+std::to_string(mpi_rank)+".qout",std::ios::out);  
    e_os.open("Efield"+std::to_string(mpi_rank)+".qout",std::ios::out);  
    d_os.open("dens"+std::to_string(mpi_rank)+".qout",std::ios::out);  

  }

  template <typename Container>
  void print(int step, const Container &f,    const Container &dens,
                       const Container &potn, const Container Efield) {

    if (step%p->large_file_intp==0) { 
      fcout << f << std::endl;
    }
    if (step%p->small_file_intp==0) { 
      d_os << dens << std::endl;
      p_os << potn << std::endl;
      e_os << Efield << std::endl;
    }

    for (auto& command: p->commands) 
      if (step>command.second.start && step%command.second.intv==0)
        system(command.second.command.c_str());
    
  }


  ~Probe() noexcept {
    fcout.close();
    p_os.close();
    e_os.close();
    d_os.close();
  }
};


} // namespace quakins
