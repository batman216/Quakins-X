#ifndef _TIMER_H_
#define _TIMER_H_
#include <chrono>
#include <string>
#include <iostream>

#include <mpi.h>

class Timer {
		
	const int mpi_rank = 0;
	std::chrono::time_point<
	          std::chrono::high_resolution_clock
	          > time1, time2, time3;
public:
	Timer();
	Timer(int);
	
	void tick(std::string message);
	void tock();
	void tock(MPI_Comm);

};



#endif /* _TIMER_H_ */
