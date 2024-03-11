
template <typename val_type, int dim>
struct TestShape; 

template <typename val_type, int dim>
struct TestShapePacket {
  val_type x10,x20,x1w,x2w,vd1, vd2, vth;

  TestShapePacket() {
    
    std::ifstream input_file(INPUT_FILE);
    auto the_map = readBox(input_file, "test_shape");
    assign(x10, "x10", the_map);
    assign(x20, "x20", the_map);
    assign(x2w, "x1w", the_map);
    assign(x2w, "x2w", the_map);
    assign(vth, "vth", the_map);
    assign(vd1, "vd1", the_map);
    assign(vd2, "vd2", the_map);
    input_file.close();

  }
};

template <typename val_type, int dim>
struct shape_packet_traits<val_type,dim,TestShape>{
  using packet = TestShapePacket<val_type,dim>;
};

template <typename val_type>
struct TestShape<val_type,4> { 

  using val_XV_t = std::array<val_type,4>;
  using Packet   = TestShapePacket<val_type,4>;

  __host__ __device__
  static val_type shape(const val_XV_t& z, Packet p) {

    auto bunch = []__host__ __device__ (val_type x1, val_type x2,
                                        val_type x10,val_type x20,
                                        val_type x1w,val_type x2w) {
      return exp(-pow((x1-x10)/x2w,2)/2-pow((x2-x20)/x2w,2)/2); 
    };

    auto f = []__host__ __device__ (val_type v1,  val_type v2,
                                    val_type vd1, val_type vd2,
                                    val_type vt) {
      return exp(-pow((v1-vd1)/vt,2)/2-pow((v2-vd2)/vt,2)/2)/(2.*M_PI*vt*vt); 
    };

    return f(z[0],z[1],p.vd1,p.vd2,p.vth) * bunch(z[2],z[3],p.x10,p.x20,p.x1w,p.x2w);

  }

};


