#include "WorkItem.h"


WorkItem::WorkItem(const std::function<void()> work, const unsigned int priority)
: work_{work}
, priority_{priority}
{

}


unsigned int WorkItem::getPriority() const {
  return priority_;
}


void WorkItem::dig() const {
  work_();
}


