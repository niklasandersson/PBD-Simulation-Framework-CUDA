#ifndef GLOBAL_DEFER_CALL_EVENT_H
#define GLOBAL_DEFER_CALL_EVENT_H

#include <mutex>
#include <utility>
#include <tuple>

#include "event/Global_Event_Base.h"
#include "event/Direct_Call_Event.h"


template<typename T> 
class Global_Defer_Call_Event;

template<class Ret, class... Args>
class Global_Defer_Call_Event<Ret(Args...)> : public Global_Event_Base, public Direct_Call_Event<Ret(Args...)> {
  
public:

  void operator()(Args... args) {
    std::tuple<Args...> arguments{std::forward<Args>(args)...};
    global_calls_lock_.lock();
    global_defered_calls_.push_back( 
      [this, arguments]() {
        if( this )
          this->make_calls(generate<sizeof...(Args)>::sequence(), arguments);
      }
    );
    global_calls_lock_.unlock();
  }

private:
  template<int ...>
  struct seq { };

  template<int N, int ...S>
  struct generate : generate<N-1, N-1, S...> { };

  template<int ...S>
  struct generate<0, S...> {
    typedef seq<S...> sequence;
  };

  template<int ...S>
  void make_calls(seq<S...>, std::tuple<Args...> arguments) {
    Direct_Call_Event<Ret(Args...)>::operator()(std::forward<Args>(std::get<S>(arguments))...);
  }

};


#endif // GLOBAL_DEFER_CALL_EVENT_H