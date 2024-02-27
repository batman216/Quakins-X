#pragma once

template <typename idx_type, typename val_type>
struct Packet_quantum<idx_type,val_type,1> {

  val_type hbar, dt, Lv, Lx; 
  idx_type nv, nvbd, nxloc, nxbd, n_chunk;

  int mpi_rank,mpi_size;
  
  val_type nx;
};



template <typename idx_type, typename val_type>
QuantumSplittingShift<idx_type,val_type,1>::QuantumSplittingShift(Packet p): p(p) {

  this->p.nx = p.nxloc*p.mpi_size; 
  val_type Dl  = 2.0*M_PI/p.Lv;
  val_type nu4 = 1e-2;

  phase.resize(p.n_chunk/2+1);
  hypercollision.resize(p.n_chunk/2+1);

  for (int i=0; i<p.n_chunk/2+1; i++)
    hypercollision[i] = exp(-nu4*pow(static_cast<val_type>(i)*Dl,4)*p.dt);;


  cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaMallocArray(&phi_tex,&channelDesc,this->p.nx,1);

  cudaResourceDesc resDesc{};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = phi_tex;

  cudaTextureDesc texDesc{};
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;

  /// the addressMode is valid only if this = 1
  texDesc.normalizedCoords = 1;

  // Create texture object
  cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL);
      
}

template <typename idx_type, typename val_type>
template <typename Container>
void QuantumSplittingShift<idx_type,val_type,1>::prepare(Container &con) {

  auto devPtr = thrust::raw_pointer_cast(con.data());
  cudaMemcpyToArray(phi_tex, 0, 0, devPtr, 
                    p.nx*sizeof(float), cudaMemcpyDeviceToDevice);
 
  int nfft = p.nv+2*p.nvbd;
  int nc = nfft/2+1, nr = nfft;
  int nbatch = p.nxloc+2*p.nxbd;
  fft_many_args<1> fmany{{nfft},{nr},1,nr,{nc},1,nc,nbatch},
                   imany{{nfft},{nc},1,nc,{nr},1,nr,nbatch};

  this->fft = new FFT<idx_type,val_type,1>(fmany,imany);

}


template<typename val_type> struct fft_complex_traist;

template<> struct fft_complex_traist<float> {
  using value_type = cufftComplex;
};

template<> struct fft_complex_traist<double> {
  using value_type = cufftDoubleComplex;
};

/// calculate f = f*exp(i*phase) in complex plane
struct exp_evolve {

  template<typename r_type,typename c_type> __device__
  c_type operator()(c_type val,r_type phase) {
    c_type buffer;
    buffer.x = val.x*cos(phase) - val.y*sin(phase); 
    buffer.y = val.y*cos(phase) + val.x*sin(phase); 
    return buffer; 

  }

};

template <typename idx_type, typename val_type>
template <typename Container>
void QuantumSplittingShift<idx_type,val_type,1>::
advance(Container& con1, Container& con2) {

  using thrust::transform;
  auto c_ptr = reinterpret_cast<cufftComplex*>(
                  thrust::raw_pointer_cast(con2.data()));

  cudaTextureObject_t tex = tex_obj;

  fft->forward(con1.begin(),con2.begin());

  val_type qDt = p.dt/p.hbar;
  val_type qDl = p.hbar*M_PI/(p.Lx*p.Lv);
  val_type Dl  = 2.0*M_PI/p.Lv;
  val_type X; /// normalized X
    
  idx_type ix, n_chunk = p.n_chunk/2+1;
  auto i_begin = thrust::make_counting_iterator((idx_type)0);
  for (idx_type i = p.nxbd; i<p.nxbd+p.nxloc; i++) {

    ix = i - p.nxbd + p.mpi_rank*p.nxloc;
    /// normalized position
    X = static_cast<val_type>(ix)/p.nx;

    /// essence of the quantum mechanical coupling!
    transform(i_begin,i_begin+n_chunk,
              hypercollision.begin(), phase.begin(),
              [tex,X,qDt,qDl] __device__
              (const val_type l, const val_type damp)->val_type
              { return damp*qDt*(tex1D<float>(tex,(X-l*qDl))
                                -tex1D<float>(tex,(X+l*qDl))); });

    transform(thrust::device,
              c_ptr+n_chunk*i,
              c_ptr+n_chunk*i+n_chunk, phase.begin(), 
              c_ptr+n_chunk*i, exp_evolve());

  }
  
  fft->inverse(con2.begin(),con1.begin());
  val_type norm = static_cast<val_type>(p.nv+p.nvbd*2);
  //thrust::for_each(con1.begin(),con1.end(),[norm]__device__(val_type& val){val/=norm;});
}  

