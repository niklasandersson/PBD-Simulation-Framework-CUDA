#include "AllTimers.h"


AllTimers::~AllTimers() {
  std::sort(timerData_.begin(), timerData_.end(), [](const TimerData& timerDataEntry1, const TimerData& timerDataEntry2) {
    return timerDataEntry1.time_ > timerDataEntry2.time_;
  });

  for(auto& timerDataEntry : timerData_) {
    // std::cout << timerDataEntry.name_ << " | " << timerDataEntry.time_ / timerDataEntry.numberOfUpdates_ << " us" << std::endl;
      std::cout << timerDataEntry.name_ << " | ttl: " << timerDataEntry.time_ << " us" 
                                        << " | avg: " << timerDataEntry.time_ / timerDataEntry.numberOfUpdates_ << " us" 
                                        << " | n: " << timerDataEntry.numberOfUpdates_ << std::endl;
  }
}


void AllTimers::add(TimerData timerDataEntry) {
  timerData_.push_back(timerDataEntry);
}