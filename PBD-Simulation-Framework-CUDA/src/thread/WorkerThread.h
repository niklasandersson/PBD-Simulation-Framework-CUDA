#ifndef WORKERTHREAD_H
#define WORKERTHREAD_H

#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>

#include "ThreadPool.h"
#include "WorkItem.h"


class ThreadPool;

class WorkerThread {

public:
  WorkerThread(const unsigned int id, ThreadPool* threadPool);

  void run();
  void stop();

protected:

private:
  const unsigned int id_;
  ThreadPool* threadPool_;

  std::atomic<bool> run_;
  std::thread workThread_;

};


#endif //WORKERTHREAD_H