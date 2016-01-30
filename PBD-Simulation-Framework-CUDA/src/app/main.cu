#include <iostream>

#include "parser/Config.h"
#include "console/Console.h"
#include "Engine.h"


Config& getConfig(int argc, const char* argv[]);
void printApplicationInfo(); 


int main(int argc, const char* argv[]) {
  Config& config = getConfig(argc, argv);

  if( config.getValue<bool>("Application.displayApplicationInfo") ) {
    printApplicationInfo();
  }  

  Engine engine;

  std::exit(EXIT_SUCCESS);
}


Config& getConfig(int argc, const char* argv[]) {
  try {
    if( argc == 2 ) {
      return Config::getInstance(argv[1]);
    }
  } catch( ... ) {
    return Config::getInstance();
  }
  return Config::getInstance();
}


void printApplicationInfo() {
  Config& config = Config::getInstance();

  const std::string name = config.getValue<std::string>("Application.name");
  const float version = config.getValue<float>("Application.version");
  const std::string* authors = config.getArray<3, std::string>("Application.authors");
  const std::string license = config.getValue<std::string>("Application.license");

  std::cout << "------------------------------------------------------------------------" << std::endl;
  std::cout << "Name: " << name << std::endl;
  std::cout << "Version: " << version << std::endl;
  std::cout << "Authors: " << authors[0] << ", " << authors[1] << " & " << authors[2] << std::endl;
  std::cout << "License: " << license << std::endl;
  std::cout << "------------------------------------------------------------------------" << std::endl;

  delete[] authors;
}
