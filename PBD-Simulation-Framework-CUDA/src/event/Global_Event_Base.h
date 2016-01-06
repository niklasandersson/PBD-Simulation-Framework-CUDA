#ifndef GLOBAL_EVENT_BASE_H
#define GLOBAL_EVENT_BASE_H

#include <vector>
#include <algorithm>
#include <mutex>
#include <functional>


class Global_Event_Base {

public:
  static void execute_calls();

protected:
  static std::mutex global_calls_lock_;
  static std::vector< std::function<void(void)> > global_defered_calls_;

};


#endif // GLOBAL_EVENT_BASE_H