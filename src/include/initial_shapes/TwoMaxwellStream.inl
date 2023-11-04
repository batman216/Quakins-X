

template <typename val_type, int dim>
struct TwoMaxwellStream; 


template <typename val_type, int dim>
struct TwoMaxwellStreamPacket {

  val_type vth1,vth2,vdf1,vdf2,Amp1,Amp2, ptb, L, kn;

  TwoMaxwellStreamPacket() {
    
    std::ifstream input_file(INPUT_FILE);
    auto the_map = readBox(input_file, "two_Maxwell");
    assign(vth1, "vthermal1", the_map);
    assign(vth2, "vthermal2", the_map);
    assign(vdf1, "vdrift1", the_map);
    assign(vdf2, "vdrift2", the_map);
    assign(Amp1, "Amplitude1", the_map);
    assign(Amp2, "Amplitude2", the_map);
    assign(ptb, "ptb", the_map);
    assign(L, "wave_length", the_map);
    assign(kn, "wave_number_norm", the_map);
    input_file.close();

  }

};

template <typename val_type, int dim>
struct shape_packet_traits<val_type,dim,TwoMaxwellStream>{
  using packet = TwoMaxwellStreamPacket<val_type,dim>;
};



template <typename val_type>
struct TwoMaxwellStream<val_type,2> { 

  using val_XV_t = std::array<val_type,2>;
  using Packet   = TwoMaxwellStreamPacket<val_type,2>;

  __host__ __device__
  static val_type shape(const val_XV_t& z,Packet p) {

    auto f = []__host__ __device__ (val_type v,
                                    val_type vd,
                                    val_type vt) {
      return expf(-powf((v-vd)/vt,2)/2)/sqrtf(2.*M_PI*vt*vt); 
    };

    return (p.Amp1*f(z[0], p.vdf1, p.vth1)+p.Amp2*f(z[0],p.vdf2,p.vth2))
             *(1.0 + p.ptb*cosf(2.0*M_PI/p.L*p.kn*z[1]));

  }

};


