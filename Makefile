EXE = quakins
CXX = nvcc

CUDA_VERSION ?= 12.0
SDK_VERSION  ?= 23.3

FFTW_HOME ?= 
EIGEN_HOME ?= /home/batman/eigen-3.4.0
SDK_HOME  ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${SDK_VERSION}
OMP_HOME  ?= ${SDK_HOME}/compilers
NCCL_HOME ?= ${SDK_HOME}/comm_libs/${CUDA_VERSION}/nccl
MPI_HOME  ?= ${SDK_HOME}/comm_libs/hpcx/hpcx-2.14/ompi

NVCFlAG = --extended-lambda --expt-relaxed-constexpr 
CXXFLAG = -std=c++20 -Xcompiler -fopenmp 

INCFLAG = -I${NCCL_HOME}/include -I${MPI_HOME}/include \
					-I${OMP_HOME}/include -I${FFTW_HOME}/include -I${EIGEN_HOME}
LIBFLAG = -L${NCCL_HOME}/lib -lnccl -L${MPI_HOME}/lib -lmpi -L${OMP_HOME}/lib -lomp \
          -L${SDK_HOME}/math_libs/${CUDA_VERSION}/targets/x86_64-linux/lib          \
					-L${FFTW_HOME}/lib \
          -lopen-rte -lopen-pal -lcufft -lcusolver -lfftw3f -lm
	
ifeq ($(mode),debug)
	CXXFLAG += -g
endif

SRC = src
BLD = bld
RUN = run

INPUT = quakins.input

$(shell mkdir -p ${BLD})

MACRO = -D ANYHOW

CPP = ${wildcard ${SRC}/*.cpp}
CU  = ${wildcard ${SRC}/*.cu}
HPP = ${wildcard ${SRC}/include/*.hpp}

CPPOBJ = ${patsubst ${SRC}/%.cpp,${BLD}/%.o,${CPP}}

MAIN = main_mpi

MAINCU = ${SRC}/${MAIN}.cu
MAINO = ${BLD}/${MAIN}.o

${BLD}/${EXE}: ${MAINO} ${CPPOBJ}
	${CXX} $^ ${NVCFlAG} ${LIBFLAG} -o $@

${MAINO}: ${MAINCU}
	${CXX} ${MACRO} ${CXXFLAG} ${INCFLAG} ${NVCFlAG} -c $< -o $@

${BLD}/%.o: ${SRC}/%.cpp 
	${CXX} ${MACRO} ${CXXFLAG} ${INCFLAG} ${NVCFlAG} -c $< -o $@

mpirun: ${BLD}/${EXE} ${INPUT}
	mkdir -p ${RUN} && cp $^ ${RUN} && cd ${RUN} && mpirun -n 2 ./${EXE}

run: ${BLD}/${EXE} ${INPUT}
	mkdir -p ${RUN} && cp $^ ${RUN} && cd ${RUN} && ./${EXE}

clean:
	rm -rf ${BLD}

show:
	echo ${CPP} ${OBJ}
