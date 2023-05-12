EXE = quakins
CXX = nvcc

NVCFlAG = --extended-lambda
CXXFLAG = -std=c++20 -Xcompiler -fopenmp
LDFLAG  = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nccl/include           \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nccl/lib               \
          -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/hpcx/hpcx-2.14/ompi/include \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/hpcx/hpcx-2.14/ompi/lib     \
          -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/include                     \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/lib                         \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/hpcx/hpcx-2.14/ompi/include \
          -lmpi -lopen-rte -lopen-pal -lomp -lnccl
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

CUOBJ = ${patsubst ${SRC}/%.cu,${BLD}/%.o,${CU}}
CPPOBJ = ${patsubst ${SRC}/%.cpp,${BLD}/%.o,${CPP}}


${BLD}/${EXE}: ${CUOBJ} ${CPPOBJ}
	${CXX} $^ ${NVCFlAG} ${LDFLAG} -o $@

${BLD}/%.o: ${SRC}/%.cu 
	${CXX} ${CXXFLAG} ${LDFLAG} ${NVCFlAG} -c $< -o $@

${BLD}/%.o: ${SRC}/%.cpp 
	${CXX} ${CXXFLAG} ${LDFLAG} ${NVCFlAG} -c $< -o $@

run: ${BLD}/${EXE} ${INPUT}
	mkdir -p ${RUN} && cp $^ ${RUN} && cd ${RUN} && ./${EXE}

clean:
	rm -rf ${BLD}

show:
	echo ${CPP} ${OBJ}
