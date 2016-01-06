#ifndef LOCAL_DEFER_CALL_EVENT_H
#define LOCAL_DEFER_CALL_EVENT_H

#include <vector>
#include <algorithm>
#include <mutex>
#include <functional>
#include <utility>
#include <tuple>

#include "event/Direct_Call_Event.h"


template<typename T> 
class Local_Defer_Call_Event;

template<class Ret, class... Args>
class Local_Defer_Call_Event<Ret(Args...)> : public Direct_Call_Event<Ret(Args...)> {
  
public:

  void operator()(Args... args) {
    local_calls_lock_.lock();
    local_defered_calls_.push_back( 
      std::pair< std::function<void(Args...)>, std::tuple<Args...> > { 
                  [this](Args&&... args) {
                    if( this )
                      this->Direct_Call_Event<Ret(Args...)>::operator()(std::forward<Args>(args)...);
                  },
                  std::tuple<Args...>{std::forward<Args>(args)...}
               }
    );
    local_calls_lock_.unlock();
  }

  void execute_calls() {
    local_calls_lock_.lock();
    std::vector<std::pair<std::function<void(Args...)>, std::tuple<Args...> > > temp_calls{
      std::move(local_defered_calls_)
    };
    local_calls_lock_.unlock();

    for(const auto& pair : temp_calls) {
      call(typename generate<sizeof...(Args)>::sequence(), pair.first, pair.second);
    }
  }

private:
  mutable std::mutex local_calls_lock_;
  std::vector<std::pair<std::function<void(Args...)>, std::tuple<Args...> > > local_defered_calls_;

  template<int ...>
  struct seq { };

  template<int N, int ...S>
  struct generate : generate<N-1, N-1, S...> { };

  template<int ...S>
  struct generate<0, S...> {
    typedef seq<S...> sequence;
  };

  template<int ...S>
  void call(seq<S...>, std::function<void(Args...)> wrapper_function, std::tuple<Args...> arguments) {
    wrapper_function(std::forward<Args>(std::get<S>(arguments))...);
  }

};


#endif // LOCAL_DEFER_CALL_EVENT_H