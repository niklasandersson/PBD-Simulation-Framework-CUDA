#include "Engine.h"


Engine::Engine() 
: stop_(false)
, stopDelegate_(Delegate<void()>::from<Engine, &Engine::stop>(this))
{
  Events::stopEngine.subscribe(stopDelegate_);

  addConsoleCommands();

  initialize();
  while( !stop_.load() ) {
    run();
  }
  cleanup();

}


Engine::~Engine() {

}


void Engine::stop() {
  stop_.store(true);
}


void Engine::addConsoleCommands() {
  auto console = Console::getInstance();

  console->add("quit", [&](const char* args) {
    canvas_->set_should_close();
    Console::getInstance()->close();
    this->stop();
  });

  console->add("q", [&](const char* args) {
    Console::getInstance()->execute("quit");
  });

}


void Engine::initialize() {
  Config& config = Config::getInstance();

  canvas_ = new Canvas{config.getValue<unsigned int>("Application.OpenGL.width"),
                       config.getValue<unsigned int>("Application.OpenGL.height"),
                       config.getValue<std::string>("Application.name")};
  canvas_->initialize();
  
  simulation_ = new Simulation();
  simulation_->initialize();

  Console::getInstance()->run();

}


void Engine::run() {

  simulation_->step();

  canvas_->render();
  
  if( canvas_->should_close() ) {
    stop();
  }

}


void Engine::cleanup() {
  canvas_->cleanup();
  delete canvas_;

  simulation_->cleanup();
  delete simulation_;

  Console::getInstance()->close();
}
