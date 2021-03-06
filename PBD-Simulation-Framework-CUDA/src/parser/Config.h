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
  static Config& getInstance(const std::string file = "config.txt") {
    static Config instance(file);
    return instance;
  }

  ~Config() = default;

  template<typename T, typename S> 
  T getValue(S arg) {
    std::vector<std::string> path = split(arg);
    return parser_.getValueImpl<T>(path);
  }

  template<typename T, typename S> 
  void setValue(T value, S arg) {
    std::vector<std::string> path = split(arg);
    parser_.setValueImpl(value, path);
  }

  template<typename T, typename S1, typename... S2>
  T getValue(S1 arg, S2... args) {
    return parser_.getValue<T>(arg, args...);
  }

  template<typename S> 
  std::vector<std::string> getDefines(S arg) {
    std::vector<std::string> path = split(arg);
    return parser_.getDefinesImpl(path);
  }

  template<typename S1, typename... S2>
  std::vector<std::string> getDefines(S1 arg, S2... args) {
    return parser_.getDefines(arg, args...);
  }

  template<typename T, typename S1, typename... S2>
  void setValue(T value, S1 arg, S2... args) {
    parser_.setValue(value, arg, args...);
  }

  template<unsigned int nArgs, typename T, typename S>
  T* getArray(S arg) {
    std::vector<std::string> path = split(arg);
    return parser_.getArrayImpl<nArgs, T>(path);
  }
  
  template<typename T, typename S>
  void setArray(const std::vector<T> values, S arg) {
    std::vector<std::string> path = split(arg);
    parser_.setArrayImpl(path, values);
  }

  template<unsigned int nArgs, typename T, typename S1, typename... S2>
  T* getArray(S1 arg, S2... args) {
    return parser_.getArray<nArgs, T>(arg, args...);
  }

  template<typename T, typename S1, typename... S2>
  void setArray(const std::vector<T> values, S1 arg, S2... args) {
    parser_.setArray<T>(values, arg, args...);
  }

  void load(const std::string file) {
    pathAndName_ = file;
    parser_.parseFile(pathAndName_);
  }

  void reload() {
    parser_.parseFile(pathAndName_);
  }

  void write(const std::string file = "NOFILE") {
    if( file == "NOFILE" ) {
      parser_.write(pathAndName_);
    } else {
      parser_.write(file);
    }
  }

  std::string at() {
    return pathAndName_;
  }

protected:

private:
  Config(const std::string file) : pathAndName_(file) {
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