#include "include/Timer.hpp"

Timer::Timer() {}


void Timer::tick(std::string message) {
	time1 = std::chrono::system_clock::now();

	std::cout << message << std::flush;
}

void Timer::tock() {
			
	time2 = std::chrono::system_clock::now();
	auto int_ms = std::chrono::duration_cast<
	std::chrono::milliseconds>(time2 - time1);

	std::cout << "("+ std::to_string(int_ms.count()) +"ms)" << std::endl;

}


