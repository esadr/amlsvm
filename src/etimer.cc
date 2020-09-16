#include "etimer.h"
#include <string>

//http://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows


void ETimer::stop_timer(const std::string desc){
    double cpu_duration = (std::clock() - start_cpu_time) / (double)CLOCKS_PER_SEC;

#if timer_print
    std::cout <<"[CTime] "<< desc <<" takes " << cpu_duration << " seconds " << std::endl;
#endif
}


void ETimer::stop_timer(const std::string desc1, const std::string desc2){
    stop_timer(desc1+" "+desc2);
}

void ETimer::stop_wall_timer(const std::string desc){
    std::chrono::system_clock::time_point wcts = std::chrono::system_clock::now();
    std::chrono::duration<double> wct_duration = (std::chrono::system_clock::now() - start_wall_time);
#if timer_print
    std::cout <<"[WTime] "<< desc <<" takes " << wct_duration.count() << 
			" seconds " << std::endl;
#endif
}

void ETimer::stop_wall_timer(const std::string desc1, const std::string desc2){
    stop_wall_timer(desc1+" "+desc2);
}

