EXE = quakins
CXX = nvcc

NVCFlAG = --extended-lambda --expt-relaxed-constexpr 
CXXFLAG = -std=c++20 -Xcompiler -fopenmp
ICLFLAG = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nccl/include           \
          -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/hpcx/hpcx-2.14/ompi/include \
          -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/include                     \

LDFLAG  = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nccl/lib                 \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/hpcx/hpcx-2.14/ompi/lib       \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/lib                           \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib \
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
