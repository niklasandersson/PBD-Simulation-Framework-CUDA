#ifndef WORKITEM_H
#define WORKITEM_H

#include <functional>


class WorkItem {

public:
  WorkItem(const std::function<void()> work, const unsigned int priority = 0);

  unsigned int getPriority() const;

  void dig() const;

protected:

private:
  const unsigned int priority_;
  const std::function<void()> work_;

};


#endif // WORKITEM_H
