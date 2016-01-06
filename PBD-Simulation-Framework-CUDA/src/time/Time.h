#ifndef TIME_H
#define TIME_H

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

class AllTimers;

#include "AllTimers.h"


class Timer {

public:

  Timer(const std::string& name, const unsigned int time = 0, const unsigned int numberOfUpdates = 0);

  ~Timer();

  void update(const unsigned int time);

  std::string getName() const;

  unsigned int getTime() const;

  unsigned int getNumberOfUpdates() const;

private:
  static AllTimers allTimers_;

  const std::string name_;
  unsigned int time_;
  unsigned int numberOfUpdates_;

};


#define TIME(e) [&](const std::string& file, const unsigned int line) \
                      {  \
                        std::ostringstream timerName{}; \
                        timerName << #e " @ " << file << " " << line;  \
                        static Timer timer{timerName.str()}; \
                        auto t1 = std::chrono::high_resolution_clock::now(); \
                        e; \
                        auto t2 = std::chrono::high_resolution_clock::now(); \
                        timer.update(std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()); \
                      }(__FILE__, __LINE__)


#endif // TIME_H