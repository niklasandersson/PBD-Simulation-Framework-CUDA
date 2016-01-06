#include "ThreadPool.h"


ThreadPool::ThreadPool(const unsigned int numberOfWorkers) 
: numberOfWorkers_{numberOfWorkers}
, workerThreadsCounter_{0}
, numberOfAddedWorkItems_{0}
, numberOfFinishedWorkItems_{0}
{
  for(unsigned int i=0; i<numberOfWorkers_; i++) {
    workThreads_.push_back(new WorkerThread{workerThreadsCounter_++, this});
    workThreads_[i]->run();
  }
}


ThreadPool::~ThreadPool() {
  clearThreads();

  for(unsigned int i=0; i<workThreads_.size(); i++) {
    delete workThreads_[i];
  }
}


void ThreadPool::clearThreads() {
  setNumberOfWorkers(0);
}


void ThreadPool::clearWorkItems() {
  WorkItem* workItem = nullptr;
  std::lock_guard<std::mutex> guardian(queueLock_);
  while( !queue_.empty() ) {
    workItem = queue_.top();
    queue_.pop();
    delete workItem;
  }
}


void ThreadPool::setNumberOfWorkers(const unsigned int numberOfWorkers) {

  if( numberOfWorkers_ >  numberOfWorkers ) {
    const unsigned int numberOfWorkersToRemove = numberOfWorkers_ - numberOfWorkers;

    for(unsigned int i=0; i<numberOfWorkersToRemove; i++) {
      const unsigned int workThread = workThreads_.size()-1;
      workThreads_[workThread]->stop();
      workThreads_.erase(workThreads_.begin() + workThread);
    }

  } else if( numberOfWorkers_ <  numberOfWorkers ) {
    const unsigned int numberOfWorkersToAdd = numberOfWorkers - numberOfWorkers_;

    for(unsigned int i=0; i<numberOfWorkersToAdd; i++) {
      workThreads_.push_back(new WorkerThread{workerThreadsCounter_++, this});
      workThreads_[workThreads_.size()-1]->run();
    }
  }

  numberOfWorkers_ = numberOfWorkers;

}


unsigned int ThreadPool::getNumberOfWorkers() const {
  return numberOfWorkers_;
}


void ThreadPool::add(WorkItem* workItem) {
  numberLock_.lock();
  numberOfAddedWorkItems_++;
  numberLock_.unlock();

  queueLock_.lock();
  queue_.push(workItem);
  queueLock_.unlock();
}


WorkItem* ThreadPool::pop() {
  WorkItem* workItem = nullptr;
  std::lock_guard<std::mutex> guardian(queueLock_);
  if( !queue_.empty() ) {
    workItem = queue_.top();
    queue_.pop();
  }
  return workItem;
}


void ThreadPool::wait() {

  while( true ) {

    numberLock_.lock();
    if( numberOfFinishedWorkItems_ == numberOfAddedWorkItems_ ) {
      numberLock_.unlock();
      return;
    } 
    numberLock_.unlock();

    WorkItem* workItem = pop();

    if( workItem != nullptr ) {
      workItem->dig();
      workerFinsihedJob();
      delete workItem;
    } 

  }

} 


void ThreadPool::workerFinsihedJob() {
  std::lock_guard<std::mutex> guardian(numberLock_);
  numberOfFinishedWorkItems_++;
}
