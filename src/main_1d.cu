
#include <iostream>
#include <mpi.h>

#include "include/initialization.hpp"
#include "include/PhaseSpaceInitialization.hpp"
#include "include/Timer.hpp"
#include "include/ReorderCopy.hpp"
#include "include/FreeStream.hpp"
#include "include/Boundary.hpp"
#include "include/Integrator.hpp"
#include "include/PoissonSolver.hpp"
#include "include/util.hpp"
#include "include/ParallelCommunicator.hpp"
#include "include/InsertGhost.hpp"
#include "include/fftOutPlace.hpp"
#include "include/Slice.hpp"
#include "include/TestParticle.hpp"

using Nums = std::size_t;
using Real = float;


constexpr Nums dim = 2;

int main(int argc, char* argv[]) {


  Parameters<Nums,Real,dim> *p = 
             new Parameters<Nums,Real,dim>;
  try { quakins::init(p, mpi_rank); }
  catch (std::invalid_argument& e) 
  {
    std::cerr << e.what() << std::endl;
#ifndef ANYHOW
    return -1;
#endif
  }

  Nums nx=p->n[1];
  Nums nxbd=p->n_ghost[1];
  Nums nxtot=p->n_all[1];
  Nums nv=p->n[0];
  Nums nxall = nxtot;
  Nums dens_size = nxtot;
  
  Real vmin=p->low_bound[0], vmax=p->up_bound[0];
  Real xmin=p->low_bound[1], xmax=p->up_bound[1];

  thrust::device_vector<Real> 
    f_e(p->n_1d_per_dev), f_e_buff(p->n_1d_per_dev);
  thrust::device_vector<Real> 
    intg_buff(nxall*nv2), dens_e(nxall), p_energy(nxall), v2(nxall);
  std::array<thrust::device_vector<Real>,dim/2> E;
  for (int i=0; i<dim/2; i++) E[i].resize(nx);

  thrust::device_vector<Real> 
    dens_e_all(nxtot), dens_e_all_buff(nxtot), pote_all_buf(nx),pote_all(nx),
    pote_all_tot(nxtot), p_energy_all(nxtot), v2_all(nxtot);
  thrust::host_vector<Real> _dens_e_all(nx), _pote_all(nx);

  std::array<Nums,2> order1 = {0,1},  order2 = {1,0};

  
  quakins::ReorderCopy<Nums,Real,dim> copy1(n_now_1,order1);
  thrust::gather(order1.begin(),order1.end(),
                 n_now_1.begin(), n_now_2.begin());
  quakins::ReorderCopy<Nums,Real,dim> copy2(n_now_2,order2);
  thrust::gather(order2.begin(),order2.end(),
                 n_now_2.begin(), n_now_3.begin());
  quakins::ReorderCopy<Nums,Real,dim> copy3(n_now_3,order3);
  thrust::gather(order3.begin(),order3.end(),
                 n_now_3.begin(), n_now_4.begin());

  quakins::ReorderCopy<Nums,Real,dim/2> dens_copy({nxtot,nx2},{1,0});
  quakins::ReorderCopy<Nums,Real,dim/2> pot_copy({nx2,nx},{1,0});

  quakins::FreeStream<Nums,Real,dim,2,0,
    quakins::details::FluxBalanceCoordSpace> fsSolverX1(p,p->dt);
  quakins::FreeStream<Nums,Real,dim,4,0,
    quakins::details::FourierSpectrumVeloSpace> vSolver(p,p->dt);
  quakins::FreeStream<Nums,Real,dim,4,0,
    quakins::details::WignerTerm> wignerSolver(p,p->dt);

  quakins::Boundary<Nums,PeriodicBoundary>
    boundX1(nx,nxbd,nv,nxtot*nv2);

  quakins::PoissonSolver<Nums,Real,1, FFT1D> 
    poissonSolver({nx},{xmin,xmax});

  quakins::Integrator<Real> 
    integral1(nv,nx,vmin,vmax);

//--------------------------------------------------------------------
  quakins::PhaseSpaceInitialization
          <Nums,Real,dim,ShapeFunctor>  phaseSpaceInit(p);
  phaseSpaceInit(thrust::device, 
                 f_e.begin(), p->n_1d_per_dev, mpi_rank);
  integral(f_e.begin(),dens_e.begin());

  std::ofstream dout("dens_e@"+std::to_string(mpi_rank)+".qout",std::ios::out);
  std::ofstream pout("potential@"+std::to_string(mpi_rank)+".qout",std::ios::out);
  std::ofstream energy_out("p_energy@"+std::to_string(mpi_rank)+".qout",std::ios::out);
  std::ofstream v_out("v@"+std::to_string(mpi_rank)+".qout",std::ios::out);
  

  dens_copy(dens_e_all.begin(),dens_e_all.end(),dens_e_all_buff.begin());
  thrust::copy(dens_e_all_buff.begin()+nx2*nxbd,
               dens_e_all_buff.end()-nx2*nxbd,
               _dens_e_all.begin());
  poissonSolver(_dens_e_all.begin(),_dens_e_all.end(),_pote_all.begin());
 
  std::puts("--------------------------");

  /*
  std::ofstream fout("ftest@" +std::to_string(mpi_rank)+ ".qout",std::ios::out);
  fout << f_e << std::endl;
  */
  quakins::Weighter<Nums,Real,1> weighter(p,mpi_rank);
  quakins::FFT<Nums,Real,1> fft(p,mpi_rank);

  std::cout << "Main loop start..." << std::endl; 
  Real t=0.0;
  for (Nums step=0; step<p->time_step_total; step++) {


    integral(f_e.begin(),dens_e.begin());

    dens_copy(dens_e_all.begin(),dens_e_all.end(),dens_e_all_buff.begin());
    thrust::copy(dens_e_all_buff.begin()+nx2*nxbd,
                 dens_e_all_buff.end()-nx2*nxbd, _dens_e_all.begin());

    poissonSolver(_dens_e_all.begin(),_dens_e_all.end(),_pote_all.begin());

    thrust::copy(_pote_all.begin(),_pote_all.end(),pote_all_buf.begin());    

    pot_copy(pote_all_buf.begin(),pote_all_buf.end(),pote_all.begin());

    if (step%(p->dens_print_intv)==0) {
      dout << _dens_e_all << std::endl;
      pout << pote_all << std::endl;
      energy_out << p_energy_all << std::endl;
      v_out << v2_all << std::endl;
    //  slice1({60,0,60,0},f_e.begin());
     // slice2({0,0,60,50},f_e.begin());
    }
    // velocity direction push  

    fft.forward(f_e.begin(),f_e.end(),f_e_buff.begin());
#ifdef QUANTUM
    wignerSolver(f_e_buff.begin(), f_e_buff.end(), pote_all.begin(),id);
#elif CLASSIC
    vSolver(f_e_buff.begin(), f_e_buff.end(), pote_all.begin(),id);
#endif
    fft.backward(f_e_buff.begin(),f_e_buff.end(),f_e.begin());


    t += p->dt;
  }


}
  


