#include <iostream>

#include "parser/Config.h"
#include "console/Console.h"
#include "Engine.h"


void printApplicationInfo() {
  Config& config = Config::getInstance();

  std::string name = config.getValue<std::string>("Application.name");
  float version = config.getValue<float>("Application.version");
  std::string* authors = config.getArray<3, std::string>("Application.authors");
  std::string license = config.getValue<std::string>("Application.license");

  std::cout << "------------------------------------------------------------------------" << std::endl;
  std::cout << "Name: " << name << std::endl;
  std::cout << "Version: " << version << std::endl;
  std::cout << "Authors: " << authors[0] << ", " << authors[1] << " & " << authors[2] << std::endl;
  std::cout << "License: " << license << std::endl;
  std::cout << "------------------------------------------------------------------------" << std::endl;

  delete[] authors;
}


int main(int argc, const char* argv[]) {

  Config& config = Config::getInstance();

  if( config.getValue<bool>("Application.displayApplicationInfo") ) {
    printApplicationInfo();
  }  

  Engine engine;

  std::exit(EXIT_SUCCESS);

}
