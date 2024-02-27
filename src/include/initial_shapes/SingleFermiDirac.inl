


template <typename val_type, int dim>
struct SingleFermiDirac; 

template <typename val_type, int dim>
struct SingleFermiDiracPacket {
  val_type Theta,mu, vd1, vd2, ptb, L, kn;

  SingleFermiDiracPacket() {
    
    std::ifstream input_file(INPUT_FILE);
    auto the_map = readBox(input_file, "single_FermiDirac");
    assign(vd1, "vd1", the_map);
    assign(vd2, "vd2", the_map);
    assign(Theta, "Theta", the_map);
    assign(mu, "mu", the_map);
    assign(ptb, "ptb", the_map);
    assign(L, "wave_length", the_map);
    assign(kn,"wave_number_norm", the_map);
    input_file.close();

  }
};

template <typename val_type, int dim>
struct shape_packet_traits<val_type,dim,SingleFermiDirac>{
  using packet = SingleFermiDiracPacket<val_type,dim>;
};


template <typename val_type>
struct SingleFermiDirac<val_type,2> { 

  using val_XV_t = std::array<val_type,2>;
  using Packet   = SingleFermiDiracPacket<val_type,2>;

  __host__ __device__
  static val_type shape(const val_XV_t& z, Packet p) {

    val_type ptb = p.ptb, vd = p.vd1;
    val_type Theta = p.Theta, mu = p.mu;

    auto f = []__host__ __device__ (val_type v, val_type vd, 
                                    val_type Theta, val_type mu) {
      return 0.75*pow(Theta,1.5)*log(1.0 + exp((mu/Theta-pow(v-vd,2)))); 

    };

    return f(z[0],vd,Theta,mu) * (1.0+ptb*cosf(2.0*M_PI/p.L*p.kn*z[1]));

  }

};


template <typename val_type>
struct SingleFermiDirac<val_type,4> { 

  using val_XV_t = std::array<val_type,4>;
  using Packet   = SingleFermiDiracPacket<val_type,4>;

  __host__ __device__
  static val_type shape(const val_XV_t& z, Packet p) {

    val_type ptb = p.ptb;
    val_type Theta = p.Theta, mu = p.mu;

    auto f = []__host__ __device__ (val_type v1, val_type v2,
                                    val_type vd1,val_type vd2, 
                                    val_type Theta, val_type mu) {
      return 0.75*pow(Theta,1.5)*log(1.0 + exp((mu/Theta-pow(v1-vd1,2)
                                                        -pow(v2-vd2,2)))); 
    };

    return f(z[0],z[1],p.vd1,p.vd2,Theta,mu) * (1.0+ptb*cosf(2.0*M_PI/p.L*p.kn*z[3]));

  }

};


