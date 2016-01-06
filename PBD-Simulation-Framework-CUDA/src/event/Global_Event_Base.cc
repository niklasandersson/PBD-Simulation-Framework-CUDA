#include "event/Global_Event_Base.h"

std::mutex Global_Event_Base::global_calls_lock_;
std::vector< std::function<void(void)> > Global_Event_Base::global_defered_calls_;


void Global_Event_Base::execute_calls() {

  if( global_calls_lock_.try_lock() ) {

    std::vector< std::function<void(void)> > temp_calls(std::move(global_defered_calls_));

    global_calls_lock_.unlock();

    for(const auto& a_call : temp_calls) {
      a_call();
    }

  }
  
}