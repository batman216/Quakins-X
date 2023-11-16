#pragma once

#include <nccl.h>
#include <thrust/device_vector.h>

namespace quakins {


template <typename idx_type, typename val_type>
struct ParallelCommunicator {

  int mpi_rank, mpi_size, local_rank=0;

  ncclUniqueId nccl_id;
  cudaStream_t s;
  ncclComm_t comm;

  ParallelCommunicator(int mpi_rank, int mpi_size, MPI_Comm mComm)
  : mpi_size(mpi_size), mpi_rank(mpi_rank)
  {

    uint64_t host_hashs[mpi_size];
    char hostname[1024];
    getHostName(hostname, 1024);
    host_hashs[mpi_rank] = getHostHash(hostname);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,host_hashs,
                  sizeof(uint64_t), MPI_BYTE, mComm);
    for (int p=0; p<mpi_size; p++) {
      if (p==mpi_rank) break;
      if (host_hashs[p]==host_hashs[mpi_rank]) local_rank++;
    }

    if (mpi_rank==0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast((void*)&nccl_id,sizeof(nccl_id),MPI_BYTE,0,mComm);
  
    cudaSetDevice(local_rank);
    cudaStreamCreate(&s);
  
    ncclCommInitRank(&comm,mpi_size,nccl_id,mpi_rank);

  }
};

template <typename val_type> 
struct nccl_traits {
  ncclDataType_t name;

  nccl_traits() {
    if constexpr (std::is_same<val_type,float>::value) 
      name = ncclFloat;
    else if constexpr(std::is_same<val_type,double>::value)
      name = ncclDouble;
    else {}

  }
  
};


template <typename idx_type, typename val_type>
struct PhaseSpaceParallelCommute {

  ParallelCommunicator<idx_type,val_type> *para;

  nccl_traits<val_type> ncclType;
  const idx_type comm_size;
  const int l_rank, r_rank;
  thrust::device_vector<val_type> l_send_buff, l_recv_buff, 
                                  r_send_buff, r_recv_buff; 
 
  PhaseSpaceParallelCommute(idx_type comm_size, 
                            ParallelCommunicator<idx_type,val_type> *para) 
  : comm_size(comm_size), para(para),  
    r_rank(para->mpi_rank==para->mpi_size-1? 0 : para->mpi_rank+1),
    l_rank(para->mpi_rank==0? para->mpi_size-1 : para->mpi_rank-1) 
  {

    l_send_buff.resize(comm_size); l_recv_buff.resize(comm_size);
    r_send_buff.resize(comm_size); r_recv_buff.resize(comm_size); 

  }

  template <typename itor_type>
  void operator()(itor_type itor_begin, itor_type itor_end) {

    thrust::copy(itor_end-2*comm_size,itor_end-comm_size, 
                 r_send_buff.begin());
    thrust::copy(itor_begin+comm_size,itor_begin+2*comm_size, 
                 l_send_buff.begin());
    ncclGroupStart();// <--
    ncclSend(thrust::raw_pointer_cast(l_send_buff.data()),
             comm_size, ncclType.name, l_rank, para->comm, para->s); 
    ncclRecv(thrust::raw_pointer_cast(r_recv_buff.data()),
             comm_size, ncclType.name, r_rank, para->comm, para->s); 
    ncclGroupEnd();

    ncclGroupStart();// -->
    ncclSend(thrust::raw_pointer_cast(r_send_buff.data()),
             comm_size, ncclType.name, r_rank, para->comm, para->s); 
    ncclRecv(thrust::raw_pointer_cast(l_recv_buff.data()),
             comm_size, ncclType.name, l_rank, para->comm, para->s); 
    ncclGroupEnd();

    thrust::copy(l_recv_buff.begin(),l_recv_buff.end(),itor_begin);
    thrust::copy(r_recv_buff.begin(),r_recv_buff.end(),itor_end-comm_size);

    cudaStreamSynchronize(para->s);
  }
};

template <typename idx_type, typename val_type>
struct DensityAllGather {

  ParallelCommunicator<idx_type,val_type> *para;

  const idx_type dens_size;

  DensityAllGather(idx_type dens_size, 
                ParallelCommunicator<idx_type,val_type> *para) 
  : dens_size(dens_size), para(para) {}


  template <typename itor_type>
  void operator()(itor_type recv_begin, itor_type send_begin) {

    ncclGroupStart();

    for (int r=0; r<para->mpi_size; r++) {
      ncclSend(thrust::raw_pointer_cast(&(*send_begin)), 
               dens_size,ncclFloat,r,para->comm,para->s);
      ncclRecv(thrust::raw_pointer_cast(&(*recv_begin))+r*dens_size, 
               dens_size,ncclFloat,r,para->comm,para->s);
    }
    ncclGroupEnd();

    cudaStreamSynchronize(para->s);
  }


};



template <typename idx_type, typename val_type>
struct DensityGather {

  ParallelCommunicator<idx_type,val_type> *para;

  const idx_type dens_size;

  DensityGather(idx_type dens_size, 
                ParallelCommunicator<idx_type,val_type> *para) 
  : dens_size(dens_size), para(para) {}


  template <typename itor_type>
  void operator()(itor_type recv_begin, itor_type send_begin) {

    ncclGroupStart();

    if (para->mpi_rank==0) {
      for (int r=0; r<para->mpi_size; r++)
        ncclRecv(thrust::raw_pointer_cast(&(*recv_begin))+r*dens_size, 
                 dens_size,ncclFloat,r,para->comm,para->s);
    }
    ncclSend(thrust::raw_pointer_cast(&(*send_begin)), 
             dens_size,ncclFloat,0,para->comm,para->s);

    ncclGroupEnd();

    cudaStreamSynchronize(para->s);
  }


};


template <typename idx_type, typename val_type>
struct PotentialBroadcast {

  ParallelCommunicator<idx_type,val_type> *para;

  const idx_type pot_size;

  PotentialBroadcast(idx_type pot_size, 
                ParallelCommunicator<idx_type,val_type> *para) 
  : pot_size(pot_size), para(para) {}


  template <typename itor_type>
  void operator()(itor_type itor_begin) {

    ncclGroupStart();

    ncclBcast(thrust::raw_pointer_cast(&(*itor_begin)), 
              pot_size,ncclFloat,0,para->comm,para->s);

    ncclGroupEnd();

    cudaStreamSynchronize(para->s);
  }


};



} // namespace quakins
