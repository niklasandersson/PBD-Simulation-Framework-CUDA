#ifndef CONFIGOBJECT_H
#define CONFIGOBJECT_H

#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <sstream>
#include <map>
#include <mutex>

#include "exception/Error.h"

class ConfigObject {

public:
  ConfigObject(const std::string& name) : name_(name) {}

  ~ConfigObject() = default;

  std::string getName() const {
    return name_;
  }

  void addChild(std::shared_ptr<ConfigObject> configObjectPtr) {
    children_[configObjectPtr->getName()] = configObjectPtr;
  }

  void removeChild(const std::string& name) {
    children_.erase(name);
  }

  std::shared_ptr<ConfigObject> getChild(const std::string& name) {
    if( children_.find(name) == std::end(children_) ) {
      throw std::runtime_error{ report_error( "Could not find child '" << name << "' in scope '" << name_ << "'" ) };
    }
    return children_[name];
  }

  void addDefine(const std::string& name, const std::string& value) {
    std::lock_guard<std::mutex> lock(definesMutex_);
    defines_[name] = value;
  }

  void removeDefine(const std::string& name) {
    std::lock_guard<std::mutex> lock(definesMutex_);
    defines_.erase(name);
  }

  std::vector<std::string> getDefines() {
    std::lock_guard<std::mutex> lock(definesMutex_);
    std::vector<std::string> rtr;
    for(auto define : defines_) {
      rtr.push_back(define.first);
    }
    return rtr;
  }

  std::string getDefine(const std::string& name) {
    std::lock_guard<std::mutex> lock(definesMutex_);
    if( defines_.find(name) == std::end(defines_) ) {
      throw std::runtime_error{ report_error( "Could not find variable '" << name << "' in scope '" << name_ << "'" ) };
    }
    return defines_[name];
  }

  void write(std::ostringstream& os) {
    std::lock_guard<std::mutex> lock(definesMutex_);
    std::map<std::string, std::string> orderedDefines{std::begin(defines_), std::end(defines_)};
    for(auto define : orderedDefines) {
      os << define.first << " = " << define.second << ";" << std::endl;
    }

    if( !defines_.empty() && !children_.empty() ) {
      os << std::endl;
    }

    std::map<std::string, std::shared_ptr<ConfigObject> > orderedChildren{std::begin(children_), std::end(children_)};
    for(auto child=std::begin(orderedChildren); child!=std::end(orderedChildren); child++) {
      os << child->first << ":" << std::endl;
      os << "{" << std::endl;
      std::ostringstream os2;
      child->second->write(os2);
      std::stringstream ss{os2.str()};
      std::string line;
      while( std::getline(ss, line) ) {
        os << "  " << line << std::endl;
      }
      os << "}" << std::endl;
      if( child != --std::end(orderedChildren) ) {
        os << std::endl;
      }
    }
  }

protected:

private:
  const std::string name_;
  std::unordered_map<std::string, std::shared_ptr<ConfigObject> > children_;

  std::mutex definesMutex_;
  std::unordered_map<std::string, std::string> defines_;

};



#endif