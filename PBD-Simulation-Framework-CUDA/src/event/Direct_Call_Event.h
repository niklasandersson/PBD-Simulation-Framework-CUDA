#ifndef DIRECT_CALL_EVENT_H
#define DIRECT_CALL_EVENT_H

#include <vector>
#include <algorithm>
#include <mutex>
#include <utility>

#include "event/Delegate.h"


template<typename T> 
class Direct_Call_Event;

template<class Ret, class... Args>
class Direct_Call_Event<Ret(Args...)> {
  
public:

  void subscribe(const Delegate<Ret(Args...)> delegate) {
    subscribers_lock_.lock();
    subscribers_.push_back(delegate);
    subscribers_lock_.unlock();
  }

  void unsubscribe(const Delegate<Ret(Args...)>& delegate) {
    subscribers_lock_.lock();
    auto it = std::find(std::begin(subscribers_), std::end(subscribers_), delegate);
    if( it != std::end(subscribers_) ) {
      subscribers_.erase(it);
    }
    subscribers_lock_.unlock();
  }

  void operator()(Args... args) const {
    subscribers_lock_.lock();   
    for(const auto& subscriber : subscribers_) {
      subscriber(std::forward<Args>(args)...);
    }
    subscribers_lock_.unlock();
  }


private:
  mutable std::mutex subscribers_lock_;
  std::vector<Delegate<Ret(Args...)> > subscribers_;

};


#endif // DIRECT_CALL_EVENT_H