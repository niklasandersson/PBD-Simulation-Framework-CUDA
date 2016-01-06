#ifndef ALLTIMERS_H
#define ALLTIMERS_H

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>


struct TimerData {

  TimerData(const std::string name,
            const unsigned int time,
            const unsigned int numberOfUpdates) 
  : name_(name)
  , time_(time)
  , numberOfUpdates_(numberOfUpdates)  
  {
  }

  ~TimerData() = default;

  std::string name_;
  unsigned int time_;
  unsigned int numberOfUpdates_;
};


class AllTimers {

public:
  AllTimers() = default;

  ~AllTimers();

  void add(TimerData timerDataEntry);

private:
  std::vector<TimerData> timerData_;

};


#endif // ALLTIMERS_H