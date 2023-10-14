#pragma once

template <typename idx_type, typename val_type>
struct Packet_quantum<idx_type,val_type,1> {

  val_type hbar, dl, dt; 
  idx_type nx, nv, dv, nvh, nxbd, nvbd, nchunk;
  idx_type nxtot,nvtot;

};



template <typename idx_type, typename val_type>
QuantumSplittingShift<idx_type,val_type,1>::QuantumSplittingShift(Packet p): p(p) {

  p.dl = 2.0*M_PI/p.nv/p.dv;
  p.nvh = p.nv/2;
  this->lambda.resize(p.nvh); 
  this->phase.resize(p.nvh);

}

template <typename idx_type, typename val_type>
template <typename Container>
void QuantumSplittingShift<idx_type,val_type,1>::prepare(Container &&con) {
  // declare and allocate memory

  // create texture object
  // create texture object: we only have to do this once!
  int n = 300;

    // Specify texture
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;

    resDesc.res.linear.devPtr = thrust::raw_pointer_cast(con.data());
    resDesc.res.linear.sizeInBytes = n*sizeof(val_type);

    // Specify texture object parameters
    cudaTextureDesc texDesc{};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL);
      

}



template <typename idx_type, typename val_type>
template <typename Container>
void QuantumSplittingShift<idx_type,val_type,1>::advance(Container&& con) {

  std::ofstream oo("te.qout");
  thrust::device_vector<val_type> buf(300);
  auto tit = thrust::make_transform_iterator(
             thrust::make_counting_iterator(0),[](int idx){return (val_type)idx/300.;});

  cudaTextureObject_t tex = tex_obj;
  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(300),
                    buf.begin(),[tex]__host__ __device__(int i)
                    {return tex1Dfetch<val_type>(tex,static_cast<val_type>((i+53)%300));});

  thrust::copy(buf.begin(),buf.end(),std::ostream_iterator<val_type>(oo," "));

}  

