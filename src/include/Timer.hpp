#ifndef _TIMER_H_
#define _TIMER_H_
#include <chrono>
#include <string>
#include <iostream>

class Timer {
		
	std::chrono::time_point<
	          std::chrono::high_resolution_clock
	          > time1, time2, time3;
public:
	Timer();
	
	void tick(std::string message);
	void tock();

};



#endif /* _TIMER_H_ */
