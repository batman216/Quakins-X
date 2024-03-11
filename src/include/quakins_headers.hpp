#ifndef _QUAKINS_HEADERS_
#define _QUAKINS_HEADERS_

#include <iostream>
#include <array>
#include <mpi.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "Parameters.hpp"
#include "PhaseSpaceInitialization.hpp"
#include "ParallelCommunicator.hpp"
#include "Integrator.hpp"
#include "PoissonSolver.hpp"
#include "SplittingShift.hpp"
#include "PermutationCopy.hpp"
#include "Boundary.hpp"
#include "gizmos.hpp"
#include "diagnosis.hpp"
#include "QuantumSplittingShift.hpp"
#endif /* _QUAKINS_HEADERS_ */

