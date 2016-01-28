#ifndef ENGINE_H
#define ENGINE_H

#include <atomic>

#include "parser/Config.h"
#include "console/Console.h"
#include "event/Events.h"
#include "Canvas.h"
#include "Simulation.h"


class Engine {

public:
  Engine();
  ~Engine() = default;

  void stop();

protected:

private:
  std::atomic<bool> stop_;
  Delegate<void()> stopDelegate_;

  Canvas* canvas_;
  Simulation* simulation_;

  void initialize();
  void addConsoleCommands();
  void cleanup();
  inline void run();

};


#endif // ENGINE_H