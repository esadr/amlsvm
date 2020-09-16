#ifndef ETIMER_H
#define ETIMER_H

#include <ctime>
#include <chrono>
#include <iostream>
#include "config_logs.h"

class ETimer{
private:
    std::clock_t start_cpu_time;
    std::chrono::system_clock::time_point start_wall_time;
public:

    ETimer(){
  	  start_cpu_time = std::clock(); 
	    start_wall_time = std::chrono::system_clock::now();
    }

    void stop_timer(const std::string desc);
    void stop_timer(const std::string desc1, const std::string desc2);
    void stop_wall_timer(const std::string desc);
    void stop_wall_timer(const std::string desc1, const std::string desc2);
};

#endif // ETIMER_H
