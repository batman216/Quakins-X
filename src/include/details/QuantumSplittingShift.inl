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

  phase.resize(p.n_chunk/2+1);
  hypercollision.resize(p.n_chunk/2+1);

  qDt = p.dt/p.hbar;
  qDl = p.hbar*M_PI/(p.Lx*p.Lv);
  Dl  = 2.0*M_PI/p.Lv;

  val_type nu4 = 0;
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

  /// the addressMode is valid only if the following = 1
  texDesc.normalizedCoords = 1;

  /// Create texture object
  cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL);

  /// prepare fft
  int nfft = p.nv+2*p.nvbd;
  int nc = nfft/2+1, nr = nfft;
  int nbatch = p.nxloc+2*p.nxbd;
  fft_many_args<1> fmany{{nfft},{nr},1,nr,{nc},1,nc,nbatch},
                   imany{{nfft},{nc},1,nc,{nr},1,nr,nbatch};
  this->fft = new FFT<idx_type,val_type,1>(fmany,imany);

}

template <typename idx_type, typename val_type>
template <typename Container>
void QuantumSplittingShift<idx_type,val_type,1>::prepare(Container &con) {

  auto devPtr = thrust::raw_pointer_cast(con.data());
  cudaMemcpyToArray(phi_tex, 0, 0, devPtr,
                    p.nx*sizeof(float), cudaMemcpyDeviceToDevice);
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
template<typename r_type>
struct Re {

  template<typename c_type> __device__
  r_type operator()(c_type val) {
    return val.x;
  }
};
template<typename r_type>
struct Im {

  template<typename c_type> __device__
  r_type operator()(c_type val) {
    return val.y;
  }
};



template <typename idx_type, typename val_type>
template <typename Container>
void QuantumSplittingShift<idx_type,val_type,1>::
advance(Container& con1, Container& con2) {

  using thrust::transform;
  auto c_ptr = reinterpret_cast<cufftComplex*>(
               thrust::raw_pointer_cast(con2.data()));

  val_type X; /// normalized X
  
  cudaTextureObject_t tex = tex_obj;
  val_type qDt_ = this->qDt, qDl_ = this->qDl; 
  /// 这几个对象都是成员函数，但是device上无法对host上的this解引用，
  /// 这里提前从host的this里面拷贝一份出来给下面的lambda捕获列表用，
  /// 如果直接捕获[*this]会慢很多

  idx_type ix, n_chunk = p.n_chunk/2+1;

  auto i_begin = thrust::make_counting_iterator((idx_type)0); 

  fft->forward(con1.begin(),con2.begin());

  for (idx_type i = p.nxbd; i<p.nxbd+p.nxloc; i++) {
    ix = i - p.nxbd + p.mpi_rank*p.nxloc;
    /// normalized position
    X = static_cast<val_type>(ix)/p.nx;

    /// essence of the quantum mechanical coupling!
    transform(i_begin,i_begin+n_chunk,
              phase.begin(),
              [qDt_,qDl_,X,tex] __host__ __device__
              (const val_type l)->val_type
              { return qDt_*(tex1D<float>(tex,X-l*qDl_)
                           -tex1D<float>(tex,X+l*qDl_)); });

    /// ! Here con2 can only store val_type object, but we treat it
    /// ! as a cufftComplex array by pointer.
    transform(thrust::device, c_ptr+n_chunk*i, c_ptr+n_chunk*(i+1),
              phase.begin(),  c_ptr+n_chunk*i, exp_evolve());
  }
  /// normalization before ifft, 这里直接对con2迭代器操作是可以的
  /// cufftComplex实际上就是float的实部float的虚部交错存储的
  idx_type norm = p.nv+2*p.nvbd;
  thrust::for_each(con2.begin(),con2.end(),[norm]__device__
                   (val_type& val){ val/=norm; });

  fft->inverse(con2.begin(),con1.begin());

}

