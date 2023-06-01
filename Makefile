EXE = quakins
CXX = nvcc

SDK_HOME ?= /opt/nvidia/hpc_sdk/Linux_x86_64/23.3
CUDA_VERSION ?= 12.0
SDK_VERSION ?= 23.3

NVCFlAG = --extended-lambda --expt-relaxed-constexpr 
CXXFLAG = -std=c++20 -Xcompiler -fopenmp
ICLFLAG = -I${SDK_HOME}/comm_libs/${CUDA_VERSION}/nccl/include             \
          -I${SDK_HOME}/comm_libs/hpcx/hpcx-2.14/ompi/include              \
          -I${SDK_HOME}/compilers/include                                  \

LDFLAG  = -L${SDK_HOME}/comm_libs/${CUDA_VERSION}/nccl/lib                 \
          -L${SDK_HOME}/comm_libs/hpcx/hpcx-2.14/ompi/lib                  \
          -L${SDK_HOME}/compilers/lib                                      \
          -L${SDK_HOME}/math_libs/${CUDA_VERSION}/targets/x86_64-linux/lib \
          -lmpi -lopen-rte -lopen-pal -lomp -lnccl -lcufft
ifeq ($(mode),debug)
	CXXFLAG += -g
endif

SRC = src
BLD = bld
RUN = run

INPUT = quakins.input

$(shell mkdir -p ${BLD})


CPP = ${wildcard ${SRC}/*.cpp}
CU  = ${wildcard ${SRC}/*.cu}
HPP = ${wildcard ${SRC}/include/*.hpp}

CPPOBJ = ${patsubst ${SRC}/%.cpp,${BLD}/%.o,${CPP}}

MAIN = main

MAINCU = ${SRC}/${MAIN}.cu
MAINO = ${BLD}/${MAIN}.o

${BLD}/${EXE}: ${MAINO} ${CPPOBJ}
	${CXX} $^ ${NVCFlAG} ${LDFLAG} -o $@

${MAINO}: ${MAINCU}
	${CXX} ${CXXFLAG} ${ICLFLAG} ${NVCFlAG} -c $< -o $@

${BLD}/%.o: ${SRC}/%.cpp 
	${CXX} ${CXXFLAG} ${ICLFLAG} ${NVCFlAG} -c $< -o $@

run: ${BLD}/${EXE} ${INPUT}
	mkdir -p ${RUN} && cp $^ ${RUN} && cd ${RUN} && ./${EXE}

clean:
	rm -rf ${BLD}

show:
	echo ${CPP} ${OBJ}
