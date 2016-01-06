#include "Console.h"


std::shared_ptr<Console> Console::getInstance() {
  static std::shared_ptr<Console> instance(new Console());
	return instance;
}


Console::Console()
: hasStopped_{true} 
{
	std::lock_guard<std::recursive_mutex> lock(commandItemsMutex_);
	addCommandItem(Command("ls", [&](const char* args){
    std::vector<std::string> commands;
    
		for_each(commandItems_.begin(), commandItems_.end(), [&](std::pair<std::string, Command> item) {
      commands.push_back(item.first);
  	});

    std::sort(commands.begin(), commands.end());

    for(auto& command : commands) {
      std::cout << "\t" << command << std::endl;
    }
	})); 

}


Console::~Console() {
  isRunning_ = false;
  consoleThread_.join();
}


void Console::remove(const char* name) {
  std::lock_guard<std::recursive_mutex> lock(commandItemsMutex_);
  commandItems_.erase(name);
}


void Console::execute(const char* args) const {
	std::istringstream is(args);
	std::string cmd;
	is >> cmd;

	std::lock_guard<std::recursive_mutex> lock(commandItemsMutex_);
	std::unordered_map<std::string, Command>::const_iterator it = commandItems_.find(cmd);
	
  if (it == commandItems_.end()) {
  	std::cout << cmd << ": command not found " << std::endl;
  	suggestions(cmd);
 	} else {
 		std::string rest;
		std::getline(is, rest);
 		try {
 			it->second.execute(rest.c_str());
 		} catch(...) {
 			std::cout << "ERROR: failed to execute command: " << cmd << std::endl;
 		}
 	}	
}


void Console::suggestions(std::string& cmd) const {
	const float LIKENESS = 0.5f;
	std::lock_guard<std::recursive_mutex> lock(commandItemsMutex_);
	for_each(commandItems_.begin(), commandItems_.end(), [&cmd, &LIKENESS](std::pair<std::string, Command> item) {
		if(Utils::Like(cmd, item.first) >= LIKENESS) {
			std::cout << " - perhaps you meant: " << item.first << std::endl;
		}
	});
}


void Console::add(const char* name, std::function<void(const char*)> function) {
	addCommandItem(Command(name, function));
}


void Console::addCommandItem(const Command& command) {
	std::lock_guard<std::recursive_mutex> lock(commandItemsMutex_);
	std::unordered_map<std::string, Command>::iterator it = commandItems_.find(command.getName());
  	if (it == commandItems_.end()) {
		commandItems_.insert({command.getName(), command});
  	} else {
  		std::cout << command.getName() << ": already added" << std::endl;
  	}
}


bool Console::getIsRunning() {
  return isRunning_;
}


void Console::run() {
  isRunning_ = true;

  if( !hasStopped_ ) {
    consoleThread_.join();
  }

  hasStopped_ = false;

  consoleThread_ = std::thread([&](Console* console) {

    const unsigned int defaultStartupFreezeTime = 500;
    std::this_thread::sleep_for(std::chrono::milliseconds(defaultStartupFreezeTime));

    while( console->getIsRunning() ) {
      std::string line;
      std::cout << "Console: ";
      std::getline(std::cin, line);
      if( line != "" && console->getIsRunning() ) {
        console->execute(line.c_str());
      }
    }
    hasStopped_ = true;
  }, this);
}


void Console::close() {
  isRunning_ = false;
}


