#pragma once 


#define INPUT_FILE "quakins.input"

#define DIM_X  1
#define DIM_V  1
#define DIM    DIM_X+DIM_V

#define MCW MPI_COMM_WORLD
#define __THE_FOLLOWING_CODE_ONLY_RUN_ON_RANK0__ if (mpi_rank==0) {
#define __THE_ABOVE_CODE_ONLY_RUN_ON_RANK0__ }



