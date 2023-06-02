EXE = test
CXX = nvcc

CUDA_VERSION ?= 12.0
SDK_VERSION  ?= 23.3

SDK_HOME  ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${SDK_VERSION}
OMP_HOME  ?= ${SDK_HOME}/compilers
NCCL_HOME ?= ${SDK_HOME}/comm_libs/${CUDA_VERSION}/nccl
MPI_HOME  ?= ${SDK_HOME}/comm_libs/hpcx/hpcx-2.14/ompi

NVCFlAG = --extended-lambda --expt-relaxed-constexpr 
CXXFLAG = -std=c++20 -Xcompiler -fopenmp

INCFLAG = -I${NCCL_HOME}/include -I${MPI_HOME}/include -I${OMP_HOME}/include
LIBFLAG = -L${NCCL_HOME}/lib -lnccl -L${MPI_HOME}/lib -lmpi -L${OMP_HOME}/lib -lomp \
          -L${SDK_HOME}/math_libs/${CUDA_VERSION}/targets/x86_64-linux/lib          \
          -lopen-rte -lopen-pal -lcufft 
	
ifeq ($(mode),debug)
	CXXFLAG += -g
endif

SRC = src
BLD = bld
RUN = run

INPUT = quakins.input

$(shell mkdir -p ${BLD})


MAIN = ReorderCopyTest

MAINCU = ${MAIN}.cu
CPP = ../src/*.cpp

${EXE}: ${MAINCU} ${CPP}
	${CXX} $^ ${NVCFlAG} ${INCFLAG} ${LIBFLAG} -o $@

run: ${BLD}/${EXE} ${INPUT}
	mkdir -p ${RUN} && cp $^ ${RUN} && cd ${RUN} && ./${EXE}

clean:
	rm -rf ${BLD}

show:
	echo ${CPP} ${OBJ}
