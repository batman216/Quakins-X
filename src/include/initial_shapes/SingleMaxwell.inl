

template <typename val_type, int dim>
struct SingleMaxwell; 

template <typename val_type, int dim>
struct SingleMaxwellPacket {
  val_type vd1, vd2, vth, ptb, L, kn;

  SingleMaxwellPacket() {
    
    std::ifstream input_file(INPUT_FILE);
    auto the_map = readBox(input_file, "single_Maxwell");
    assign(vth, "vth", the_map);
    assign(vd1, "vd1", the_map);
    assign(vd2, "vd2", the_map);
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


template <typename val_type>
struct SingleMaxwell<val_type,4> { 

  using val_XV_t = std::array<val_type,4>;
  using Packet   = SingleMaxwellPacket<val_type,4>;

  __host__ __device__
  static val_type shape(const val_XV_t& z, Packet p) {

    val_type vd1 = p.vd1, vd2 = p.vd2, vth = p.vth, ptb = p.ptb;

    auto f = []__host__ __device__ (val_type v1,
                                    val_type v2,
                                    val_type vd1,
                                    val_type vd2,
                                    val_type vt) {
      return exp(-pow((v1-vd1)/vt,2)/2-pow((v2-vd2)/vt,2)/2)/(2.*M_PI*vt*vt); 
    };

    return f(z[0],z[1],vd1,vd2,vth) * (1.0+ptb*cos(2.0*M_PI/p.L*p.kn*z[3]));

  }

};


