#include "WorkerThread.h"


WorkerThread::WorkerThread(const unsigned int id, ThreadPool* threadPool) 
: id_{id}
, threadPool_{threadPool}
{

}


void WorkerThread::run() {
  run_.store(true);

  workThread_= std::thread([&]() {
      
    bool wait = false;

    while( run_.load() ) {

      if( wait ) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        wait = false;
        if( !run_.load() ) {
          return;
        }
      }

      if( threadPool_ ) {
        WorkItem* workItem = threadPool_->pop();

        if( workItem != nullptr ) {
          workItem->dig();
          threadPool_->workerFinsihedJob();
          delete workItem;
        } else {
          wait = true;
        } 
      } else {
        return;
      }

    }

  });

  workThread_.detach();
}


void WorkerThread::stop() {
  run_.store(false);
}


