#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <sstream>
#include <algorithm>
#include <typeinfo>
#include <stdexcept>

#include <exception/Error.h>

#include "parser/Parser.h"
#include "parser/CommentParser.h"
#include "parser/IncludeParser.h"
#include "parser/DefineParser.h"
#include "parser/ConfigParser.h"

class Config {

public:
  static Config& getInstance() {
    static Config instance;
    return instance;
  }

  ~Config() = default;

  template<typename T, typename S> 
  T getValue(S arg) {
    std::vector<std::string> path = split(arg);
    return parser_.getValueImpl<T>(path);
  }

  template<typename T, typename S1, typename... S2>
  T getValue(S1 arg, S2... args) {
    return parser_.getValue<T>(arg, args...);
  }

  template<unsigned int nArgs, typename T, typename S>
  T* getArray(S arg) {
    std::vector<std::string> path = split(arg);
    return parser_.getArrayImpl<nArgs, T>(path);
  }

  template<unsigned int nArgs, typename T, typename S1, typename... S2>
  T* getArray(S1 arg, S2... args) {
    return parser_.getArray<nArgs, T>(arg, args...);
  }

  void reload() {
    parser_.parseFile(pathAndName_);
  }

protected:

private:
  Config() {
    pathAndName_ = "config.txt";
    parser_.addDefine("true", "1");
    parser_.addDefine("false", "0");
    parser_.parseFile(pathAndName_);
  }

  std::vector<std::string> split(const std::string& arg) {
    std::istringstream is{arg};
    std::vector<std::string> path;
    std::string temp;
    while( std::getline(is, temp, '.') ) {
      path.push_back(temp);
    }
    return path;
  }

  std::string pathAndName_;
  ConfigParser<IncludeParser<DefineParser<CommentParser<Parser> > > > parser_;

};

#endif