#include "WorkItemComparison.h"


bool WorkItemComparison::operator()(const WorkItem* first, const WorkItem* second) const {
   return first->getPriority() > second->getPriority();
}

