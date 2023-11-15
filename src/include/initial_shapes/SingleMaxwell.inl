

template <typename val_type, int dim>
struct SingleMaxwell; 

template <typename val_type, int dim>
struct SingleMaxwellPacket {
  val_type vd, vth, ptb, L, kn;

  SingleMaxwellPacket() {
    
    std::ifstream input_file(INPUT_FILE);
    auto the_map = readBox(input_file, "single_Maxwell");
    assign(vth, "vth", the_map);
    assign(vd, "vd", the_map);
    assign(ptb, "ptb", the_map);
    assign(L, "wave_length", the_map);
    assign(kn,"wave_number_norm", the_map);
    input_file.close();

  }
};

template <typename val_type, int dim>
struct shape_packet_traits<val_type,dim,SingleMaxwell>{
  using packet = SingleMaxwellPacket<val_type,dim>;
};


template <typename val_type>
struct SingleMaxwell<val_type,2> { 

  using val_XV_t = std::array<val_type,2>;
  using Packet   = SingleMaxwellPacket<val_type,2>;

  __host__ __device__
  static val_type shape(const val_XV_t& z, Packet p) {

    val_type vd = p.vd, vth = p.vth, ptb = p.ptb;

    auto f = []__host__ __device__ (val_type v,
                                    val_type vd,
                                    val_type vt) {
      return exp(-pow((v-vd)/vt,2)/2)/sqrt(2.*M_PI)/vt; 
    };

    return f(z[0],vd,vth) * (1.0+ptb*cos(2.0*M_PI/p.L*p.kn*z[1]));

  }

};



template <typename val_type, int dim>
struct SingleMaxwell_kvspace; 

template <typename val_type>
struct SingleMaxwell_kvspace<val_type,2> { 

  using val_XV_t = std::array<val_type,2>;
  using Packet   = SingleMaxwellPacket<val_type,2>;

  __host__ __device__
  static val_type shape(const val_XV_t& z, Packet p) {

    val_type vd = p.vd, vth = p.vth, ptb = p.ptb;

    auto f = []__host__ __device__ (val_type v,
                                    val_type vd,
                                    val_type vt) {
      return exp(-pow(v/vt,2)/2); 
    };

    return f(z[0],vd,vth) * (1.0+ptb*cos(2.0*M_PI/p.L*p.kn*z[1]));

  }

};


