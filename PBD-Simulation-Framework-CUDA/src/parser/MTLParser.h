#ifndef MTLPARSER_H
#define MTLPARSER_H

#include <glm/glm.hpp>

#include "parser/RecursiveParser.h"
#include "parser/OBJMaterial.h"


template<typename Parser>
class MTLParser : public RecursiveParser<Parser> {

public:
  virtual ~MTLParser() = default;

  std::vector<OBJMaterial> getMaterials() {
    return objMaterials_;
  }

  void clear() {
    objMaterials_.clear();
  }

protected:

  void setup() override {
  } 

  void cleanup() override {
  }

  std::string actualLineParsing(const std::string& onePreParsedLine) override {

    std::istringstream is{onePreParsedLine};

    std::string type;

    is >> std::ws >> type;

    // std::cout << "type = " << type << std::endl;

    if( type == "newmtl" ) {
      readNewMaterial(is);
    } else if( type == "Ka" ) {
        objMaterials_[objMaterials_.size()-1].ka_ = readColor(is);
    } else if( type == "Kd" ) {
        objMaterials_[objMaterials_.size()-1].kd_ = readColor(is);
    } else if( type == "Ks" ) {
        objMaterials_[objMaterials_.size()-1].ks_ = readColor(is);
    } else if( type == "Ke" ) {
        objMaterials_[objMaterials_.size()-1].ke_ = readColor(is);
    }

    return onePreParsedLine;
  }

private:
  std::vector<OBJMaterial> objMaterials_;

  void readNewMaterial(std::istringstream& is) {
    std::string name;
    if( is >> std::ws >> name ) {
      objMaterials_.push_back(OBJMaterial{name});
    }
  }

  glm::vec3 readColor(std::istringstream& is) {
    float r, g, b;
    is >> std::ws >> r >> std::ws >> g >> std::ws >> b;
    return glm::vec3{r, g, b};
  }

};


#endif // MTLPARSER_H