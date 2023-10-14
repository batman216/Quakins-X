#define SIGN(y) (y>=0?1:-1)

namespace quakins {

template <typename idx_type,typename val_type>
FluxBalanceMethod<idx_type,val_type>::
FluxBalanceMethod(Packet p) : p(p) {
  p.nxtot = p.nx + p.nxbd*2;
  p.nvtot = p.nv + p.nvbd*2;
  alpha.resize(p.nv); shift.resize(p.nv);
  flux.resize(p.nchunk);

}

template <typename idx_type,typename val_type>
template <typename Container>
void FluxBalanceMethod<idx_type,val_type>::prepare(Container con) {
  
  
  thrust::transform(con.begin(),con.end(),alpha.begin(),
                    [*this](val_type v){ return v * p.dt / p.dx;  });

  auto zitor = thrust::make_zip_iterator(thrust::make_tuple(alpha.begin(),shift.begin()));

  thrust::for_each(zitor,zitor+p.nv, [](auto& t){
    val_type shift_d;
    thrust::get<0>(t) = std::modf(thrust::get<0>(t),&shift_d);
    thrust::get<1>(t) = - static_cast<int>(shift_d) 
                        + (thrust::get<0>(t) >0 ? 0:1);
  });
  
}


template<typename val_type>
struct intp_intg {

  const val_type a, c1, c2;
  intp_intg(val_type a) : a(a),
                          c1((1.0-a)*(2.0-a)/6.0), 
                          c2((1.0-a)*(1.0+a)/6.0) {}

  template<typename...T>
  __host__ __device__
  val_type operator()(const thrust::tuple<T...>& t) {
     return   a*(thrust::get<1>(t) 
            +c1*(thrust::get<2>(t)-thrust::get<1>(t))
            +c2*(thrust::get<1>(t)-thrust::get<0>(t)));
  }
};


template <typename idx_type,typename val_type>
template <typename Container>
void FluxBalanceMethod<idx_type,val_type>::advance(Container con) {

  using namespace thrust::placeholders; // where _1 ... _9 are defined

  // the beginning of simulation area, without the ghost cells
  auto itor_flux = flux.begin() + p.nxbd - 1;

  int sign_v, I; bool v_is_pos; val_type a;  

  for (std::size_t i=p.nvbd; i<p.nv+p.nvbd; i++) {

    I = i - p.nvbd;
    sign_v = SIGN(alpha[I]);
    v_is_pos = sign_v==1? true:false;
    
    /// the shifted iterator pointed to the f value storge
    auto itor_left = con.begin() + i*p.nchunk;

    /// point to the leftmost value (including the ghost cells)
    auto itor_shift = itor_left +p.nxbd + shift[I] - 1;
    // nx cells indicate nx+1 faces

    a = v_is_pos? alpha[I]:1+alpha[I];

    thrust::fill(flux.begin(),flux.end(),0.); // clear the buffer

    intp_intg func(a); // prepare the interpolation algorithm
    auto zitor_shift = make_zip_iterator(thrust::make_tuple(
                         itor_shift-1, itor_shift, itor_shift+1));
    thrust::transform(zitor_shift, zitor_shift+p.nx+1, itor_flux, func);
 
    for (std::size_t k=0; k<std::abs(shift[I]); k++) 
      thrust::transform(itor_flux,itor_flux+p.nx+1,
                        itor_shift + sign_v*k + (v_is_pos?1:0),
                        itor_flux, _1+sign_v*_2);
    

    // calculate f[i](t+dt)=f[i](t) + flux[i-1/2] -flux[i+1/2]
    thrust::adjacent_difference(flux.begin(),flux.end(),flux.begin());
    thrust::transform(itor_left+p.nxbd,itor_left+p.nxbd+p.nx,
                      flux.begin()+p.nxbd,itor_left+p.nxbd, _1-_2);

  }

}


} // namespace quakins
