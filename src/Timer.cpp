#include "include/Timer.hpp"

Timer::Timer() {}
Timer::Timer(int mpi_rank):mpi_rank(mpi_rank){}


void Timer::tick(std::string message) {
	if (mpi_rank==0) {
		time1 = std::chrono::system_clock::now();

		std::cout << message << std::flush;
	}
}

void Timer::tock() {
			
	if (mpi_rank==0) {
		time2 = std::chrono::system_clock::now();
		auto int_ms = std::chrono::duration_cast<
		std::chrono::milliseconds>(time2 - time1);
		std::cout << "("+ std::to_string(int_ms.count()) +"ms)" << std::endl;
	}

}


void Timer::tock(MPI_Comm comm) {

	this->tock();
	MPI_Barrier(comm);

}


