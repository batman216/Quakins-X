


template <typename val_type, int dim>
struct SingleFermiDirac; 

template <typename val_type, int dim>
struct SingleFermiDiracPacket {
  val_type Theta,mu, vd, ptb, L, kn;

  SingleFermiDiracPacket() {
    
    std::ifstream input_file(INPUT_FILE);
    auto the_map = readBox(input_file, "single_FermiDirac");
    assign(vd, "vd", the_map);
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

    val_type ptb = p.ptb, vd = p.vd;
    val_type Theta = p.Theta, mu = p.mu;

    auto f = []__host__ __device__ (val_type v, val_type vd, 
                                    val_type Theta, val_type mu) {
      return 0.75*Theta*log(1.0 + exp((mu-pow(v-vd,2))/Theta)); 
    };

    return f(z[0],vd,Theta,mu) * (1.0+ptb*cosf(2.0*M_PI/p.L*p.kn*z[1]));

  }

};


