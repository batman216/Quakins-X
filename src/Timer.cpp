#include "include/Timer.hpp"

Timer::Timer(std::string name) :name(name) {}

Timer::Timer(int mpi_rank, std::string name)
  :name(name), mpi_rank(mpi_rank) {}


void Timer::tick(std::string message) {
  if (mpi_rank==0) {
    time1 = std::chrono::system_clock::now();

    std::cout << message << std::flush;
  }
}

void Timer::tock() {
      
  if (mpi_rank==0) {
    count++;
    time2 = std::chrono::system_clock::now();
    auto int_ms = std::chrono::duration_cast<
      std::chrono::milliseconds>(time2 - time1);
    
    dt = int_ms.count(); 
    dt_tot += dt;
    std::cout << "("+ std::to_string(dt) +"ms)" << std::endl;
  }

}


void Timer::tock(MPI_Comm comm) {

  this->tock();
  MPI_Barrier(comm);

}


Timer::~Timer() {
  if (mpi_rank==0) {

    std::string average_message; 

    if (dt_tot/count>10000) 
      average_message = std::to_string(dt_tot/count/1000) + "s on average.";
    else
      average_message = std::to_string(dt_tot/count) + "ms on average.";

    std::cout << this->name + " costs " 
      + std::to_string(dt_tot/1000) +"s in total, " + average_message << std::endl;
  }

}


