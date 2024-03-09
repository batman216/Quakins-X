

// assign value from string
template <class T>
void assign(T& val, const std::string& name, 
            std::map<std::string, std::string> input) 
{
  std::stringstream buffer(input[name]);
  buffer >> val;
}

// read box in the input file
std::map<std::string, std::string>
readBox(std::ifstream& is,std::string box_name) {

  using std::string;
  string s, key, value;
  std::map<string, string> input_map;

  getline(is, s); 
  while (s.find("@ "+box_name)==string::npos) {
    try { 
      getline(is, s);  
      if (s.find("the end")!=string::npos) { 
        throw nbe;
      }
    } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
      exit(-1);
    }
  }

  for (getline(is, s); s.find("@ "+box_name)==string::npos;
       getline(is, s)) {

    auto equ = s.find('=');

    if (equ != string::npos) {
      key = s.substr(0, equ - 1);
      key.erase(std::remove(key.begin(),
                key.end(),' '),key.end());
      value = s.substr(equ + 1, s.size());
                       value.erase(std::remove(value.begin(),
                       value.end(),' '),value.end());
      input_map[key] = value;
    }

  }
  return input_map;
}
 

template <typename idx_type, typename val_type, int dim_x, int dim_v>
Parameters<idx_type,val_type,dim_x,dim_v>::Parameters(int mpi_rank, int mpi_size, int dim_para) 
:mpi_rank(mpi_rank), mpi_size(mpi_size), dim_para(dim_para) {}



template <typename idx_type, typename val_type, int dim_x, int dim_v>
Parameters<idx_type,val_type,dim_x,dim_v>::Parameters() 
:mpi_rank(0), mpi_size(1), dim_para(1) {}



template <typename idx_type, typename val_type, int dim_x, int dim_v>
void Parameters<idx_type,val_type,dim_x,dim_v>::initial() 
{
  readTimeBox(INPUT_FILE);
  readDomainBox(INPUT_FILE);
  readQuantumBox(INPUT_FILE);
  readIOBox(INPUT_FILE);

}

template <typename idx_type, typename val_type, int dim_x, int dim_v>
void Parameters<idx_type,val_type,dim_x,dim_v>::readTimeBox(std::string filename) 
{
  std::ifstream input_file(filename);
  auto the_map = readBox(input_file, "time");

  assign(time_factor, "time_factor", the_map);
  assign(stop_at, "step_total", the_map);
  
  __THE_FOLLOWING_CODE_ONLY_RUN_ON_RANK0__
    
  std::cout << "This shot is about to run " << stop_at << " steps" << std::endl;

  __THE_ABOVE_CODE_ONLY_RUN_ON_RANK0__

  input_file.close();
}


template <typename idx_type, typename val_type, int dim_x, int dim_v>
void Parameters<idx_type,val_type,dim_x,dim_v>::readQuantumBox(std::string filename) 
{
  std::ifstream input_file(filename);
  auto the_map = readBox(input_file, "quantum");

  assign(hbar, "hbar", the_map);
  assign(degeneracy, "degeneracy", the_map);

  input_file.close();
}

template <typename idx_type, typename val_type, int dim_x, int dim_v>
void Parameters<idx_type,val_type,dim_x,dim_v>::
readDomainBox(std::string filename) {

  std::ifstream input_file(filename);
  auto the_map = readBox(input_file, "domain");

  for (int i=0; i<dim_x; i++) {
    assign(n_main_x[i], "nx"+std::to_string(i+1), the_map);
    assign(n_ghost_x[i], "nx"+std::to_string(i+1)+"_ghost", the_map);
    n_all_x[i] = n_main_x[i] + n_ghost_x[i]*2;
    n_main_x_loc[i] = n_main_x[i];
    n_all_x_loc[i] = n_all_x[i];

    assign(xmin[i], "x"+std::to_string(i+1)+"min", the_map);
    assign(xmax[i], "x"+std::to_string(i+1)+"max", the_map);
    Lx[i] = xmax[i] - xmin[i];
    dx[i] = Lx[i]/n_main_x[i];
  }

  for (int i=0; i<dim_v; i++) {
    assign(n_main_v[i], "nv"+std::to_string(i+1), the_map);
    assign(n_ghost_v[i],"nv"+std::to_string(i+1)+"_ghost", the_map);
    n_main_v_loc[i] = n_main_v[i];
    n_all_v[i] = n_main_v[i] + n_ghost_v[i]*2;
    n_all_v_loc[i] = n_all_v[i];

    assign(vmin[i], "v"+std::to_string(i+1)+"min", the_map);
    assign(vmax[i], "v"+std::to_string(i+1)+"max", the_map);
    Lv[i] = vmax[i] - vmin[i];
    dv[i] = Lv[i]/n_main_v[i];

  }
  input_file.close();

  auto VMAX_itor = std::max_element(vmax.begin(),vmax.end(),[](auto a, auto b)
                                    { return std::abs(a) < std::abs(b);});
  val_type dt_norm = dx[std::distance(vmax.begin(), VMAX_itor)]
                     /static_cast<val_type>(*VMAX_itor);

  this->time_step  = this->time_factor * dt_norm;

#ifdef PARALLEL
  n_main_x_loc[dim_para] /= mpi_size;
  n_all_x[dim_para] = n_main_x[dim_para]+2*mpi_size*n_ghost_x[dim_para];
  n_all_x_loc[dim_para] = n_all_x[dim_para]/mpi_size;
#endif

  /// the total data number of a single species
  n_whole =  std::accumulate(n_all_x.begin(),n_all_x.end(),1,std::multiplies<idx_type>())
            *std::accumulate(n_all_v.begin(),n_all_v.end(),1,std::multiplies<idx_type>());
  n_whole_main =  std::accumulate(n_main_x.begin(),n_main_x.end(),1,std::multiplies<idx_type>())
                 *std::accumulate(n_main_v.begin(),n_main_v.end(),1,std::multiplies<idx_type>());

  n_whole_loc = n_whole/mpi_size;
  n_whole_main_loc = n_whole_main/mpi_size;


  __THE_FOLLOWING_CODE_ONLY_RUN_ON_RANK0__
  
  std::cout << "dt = " << time_step << std::endl; 
  std::cout << "a single phase space fluid has " << n_whole << " numbers" << std::endl; 
  std::cout << "a single phase space fluid costs " 
  << 2*n_whole*sizeof(val_type)/1073741824.0 << "GB of memory, " 
  << 2*n_whole_loc*sizeof(val_type)/1073741824.0 << "GB for each GPU." <<std::endl;

  __THE_ABOVE_CODE_ONLY_RUN_ON_RANK0__


}


template <typename idx_type, typename val_type, int dim_x, int dim_v>
void Parameters<idx_type,val_type,dim_x,dim_v>::readIOBox(std::string filename) 
{

  std::ifstream input_file(filename);
  auto the_map = readBox(input_file, "IO_control");

  assign(large_file_intp, "large_file_intp", the_map);
  assign(small_file_intp, "small_file_intp", the_map);

  input_file.close();
  input_file.open(filename);
  this->commands = readRuntimeCommand(input_file);
  input_file.close();

}

