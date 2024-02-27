#pragma once 


#define INPUT_FILE "quakins.input"

#define MCW MPI_COMM_WORLD
#define __THE_FOLLOWING_CODE_ONLY_RUN_ON_RANK0__ if (mpi_rank==0) {
#define __THE_ABOVE_CODE_ONLY_RUN_ON_RANK0__ }



