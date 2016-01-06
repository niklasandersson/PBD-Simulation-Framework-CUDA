#include "Time.h"


AllTimers Timer::allTimers_;


Timer::Timer(const std::string& name, const unsigned int time, const unsigned int numberOfUpdates) 
: name_(name)
, time_(time)
, numberOfUpdates_(numberOfUpdates)
{
}


Timer::~Timer() {
  //std::cout << name_ << " | " << time_ / numberOfUpdates_ << " us" << std::endl;
  allTimers_.add(TimerData{name_, time_, numberOfUpdates_});
}


void Timer::update(const unsigned int time) {
  time_ += time;
  numberOfUpdates_++;
}


std::string Timer::getName() const {
  return name_;
}


unsigned int Timer::getTime() const {
  return time_;
}


unsigned int Timer::getNumberOfUpdates() const {
  return numberOfUpdates_;
}