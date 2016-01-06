#ifndef WORKITEMCOMPARISON_H
#define WORKITEMCOMPARISON_H

#include "WorkItem.h"

class WorkItemComparison
{ 

public: 
  bool operator()(const WorkItem* first, const WorkItem* second) const;

};


#endif // WORKITEMCOMPARISON_H
