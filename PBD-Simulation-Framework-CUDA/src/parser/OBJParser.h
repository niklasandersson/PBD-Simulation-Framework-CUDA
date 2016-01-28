#ifndef OBJPARSER_H
#define OBJPARSER_H

#include <glm/glm.hpp>

#include "parser/RecursiveParser.h"
#include "parser/OBJGroup.h"

#include "parser/MTLParser.h"
#include "parser/OBJMaterial.h"


template<typename Parser>
class OBJParser : public RecursiveParser<Parser> {

public:
  virtual ~OBJParser() = default;

  std::vector<float> getVertices() {
    return vertices_;
  }

  std::vector<float> getUVs() {
    return uvs_;
  }

  std::vector<float> getNormals() {
    return normals_;
  }

  std::vector<unsigned short> getVertexIndices() {
    return vertexIndices_;
  }

  std::vector<OBJGroup> getObjGroups() {
    return objGroups_;
  }

  std::vector<OBJMaterial> getObjMaterials() {
    return mtlParser_.getMaterials();
  }

  virtual void postProcessing() {
    std::vector<float> normalsCopy{normals_};
    normals_.clear();
    normals_.resize(vertexIndices_.size()*3);

    for(unsigned int i=0; i<vertexIndices_.size(); i++) {
      unsigned int index = vertexIndices_[i]*3;
      for(unsigned int j=0; j<3; j++) {
        normals_[index+j] = normalsCopy[normalIndices_[i]*3+j];
      }
    }

    // for(unsigned int i=0; i<vertexIndices_.size(); i+=3) {
    //   unsigned int i1 = vertexIndices_[i+0];
    //   unsigned int i2 = vertexIndices_[i+1];
    //   unsigned int i3 = vertexIndices_[i+2];

    //   glm::vec3 v0 = glm::vec3(vertices_[i1*3+0], vertices_[i1*3+1], vertices_[i1*3+2]);
    //   glm::vec3 v1 = glm::vec3(vertices_[i2*3+0], vertices_[i2*3+1], vertices_[i2*3+2]);
    //   glm::vec3 v2 = glm::vec3(vertices_[i3*3+0], vertices_[i3*3+1], vertices_[i3*3+2]);

    //   glm::vec3 edge1 = v2 - v0;
    //   glm::vec3 edge2 = v1 - v0;
    //   // glm::vec3 edge1 = v0 - v1;
    //   // glm::vec3 edge2 = v2 - v1;

    //   glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

    //   for(unsigned int v=0; v<3; v++) {
    //     unsigned int index = vertexIndices_[i+v]*3;
    //     normals_[index+0] = normal[0];
    //     normals_[index+1] = normal[1];
    //     normals_[index+2] = normal[2];
    //   } 
    // }

  }

  void clear() {

    mtlParser_.clear();
    
    objGroups_.clear();
    activeGroups_.clear();
    vertices_.clear();
    uvs_.clear();
    normals_.clear();
    vertexIndices_.clear();
    uvIndices_.clear();
    normalIndices_.clear();

    vertexIndicesReadOffset_ = 0;
    uvIndicesReadOffset_ = 0;
    normalIndicesReadOffset_ = 0;

  }

protected:

  void setup() override {

    if( !createdDummy_ ) {
      createDummyGroup();
      createdDummy_ = true;
    }
    
    resetForNewGroup();
  } 


  void resetForNewGroup() {
    atTheFirstFace_ = true;
    facesSet_ = false;
    facesHasVertex_ = false;
    facesHasUv_ = false;
    facesHasNormal_ = false;
    nVerticesSet_ = false;
    nUVsSet_ = false;
    nNormalsSet_ = false;
    nFacesGroupsSet_ = false;
  }


  void createDummyGroup() {
    dummyGroup_.name_ = "dummy";
    dummyGroup_.globalOffsetVertices_ = 0;
    dummyGroup_.globalOffsetUVs_ = 0;
    dummyGroup_.globalOffsetNormals_ = 0;
    dummyGroup_.globalSizeVertices_ = 0;
    dummyGroup_.globalSizeUVs_ = 0;
    dummyGroup_.globalSizeNormals_ = 0;
    objGroups_.push_back(dummyGroup_);
    activeGroups_.push_back(&objGroups_[0]);
  }

  void cleanup() override {
    vertexIndicesReadOffset_ = vertices_.size() / 3;
    uvIndicesReadOffset_ = uvs_.size() / 3;
    normalIndicesReadOffset_ = normals_.size() / 3;

    // for(auto& objgrp : objGroups_) {
    //   objgrp.print();
    // }

    // for(auto& objmtl : objMaterials_) {
    //   objmtl.print();
    // }

    // postProcessing();
    
    // std::cout << "vertices_.size(): " << vertices_.size() << std::endl;
    // std::cout << "uvs_.size(): " << uvs_.size() << std::endl;
    // std::cout << "normals_.size(): " << normals_.size() << std::endl;
    // std::cout << std::endl;
    // std::cout << "vertexIndices_.size(): " << vertexIndices_.size() << std::endl;
    // std::cout << "uvIndices_.size(): " << uvIndices_.size() << std::endl;
    // std::cout << "normalIndices_.size(): " << normalIndices_.size() << std::endl;
    // std::cout << std::endl;
    // std::cout << "facesHasVertex_: " << facesHasVertex_ << std::endl;
    // std::cout << "facesHasUv_: " << facesHasUv_ << std::endl;
    // std::cout << "facesHasNormal_: " << facesHasNormal_ << std::endl;
    // std::cout << std::endl;
    // std::cout << "nVertices_: " << nVertices_ << std::endl;
    // std::cout << "nUVs_: " << nUVs_ << std::endl;
    // std::cout << "nNormals_: " << nNormals_ << std::endl;
    // std::cout << "nFacesGroups_: " << nFacesGroups_ << std::endl;
  }


  std::string actualLineParsing(const std::string& onePreParsedLine) override {

    std::istringstream is{onePreParsedLine};

    // std::cout << onePreParsedLine << std::endl;

    std::string type;

    std::getline(is, type, ' ');

    // std::cout << "TYPE: " << type << std::endl;

    if( type == "o" ) {
      readObject(is);
    } else if( type == "g" ) {
      readGroup(is);
    } else if( type == "#" ) {
      readComment(is);
    } else if( type == "mtllib" ) {
      readMaterials(is);
    } else if( type == "usemtl" ) {
      useMaterial(is);
    } else if( type == "v" ) {
      bool nVerticesSetTemp = nVerticesSet_;
      readValues(is, vertices_, nVerticesSet_, nVertices_);
      if( !nVerticesSetTemp ) {
        for(auto activeGroup : activeGroups_) {
          activeGroup->nVerticesSet_ = nVerticesSet_;
          activeGroup->nVertices_ = nVertices_;
        }
      }
    } else if( type == "vt" ) {
      bool nUVsSetTemp = nUVsSet_;
      readValues(is, uvs_, nUVsSet_, nUVs_);
      if( !nUVsSetTemp ) {
        for(auto activeGroup : activeGroups_) {
          activeGroup->nUVsSet_ = nUVsSet_;
          activeGroup->nUVs_ = nUVs_;
        }
      }
    } else if( type == "vn" ) {
      bool nNormalsSetTemp = nNormalsSet_;
      readValues(is, normals_, nNormalsSet_, nNormals_);
      if( !nNormalsSetTemp ) {
        for(auto activeGroup : activeGroups_) {
          activeGroup->nNormalsSet_ = nNormalsSet_;
          activeGroup->nNormals_ = nNormals_;
        }
      }
    } else if( type == "f" ) {
      readFaces(is);
    } else if( type == "s" ) {
      smoothShading(is);
    }

    return onePreParsedLine;
  }


  bool existsAnotherGroupInstance(const std::string& name, OBJGroup& mostRecentGroupInstance) {
    for(int i=objGroups_.size()-1; i>=0; i--) {
      if( objGroups_[i].name_ == name ) {
        mostRecentGroupInstance = objGroups_[i];
        return true;
      }
    }
    return false;
  }

  void readObject(std::istringstream& is) {
    // o object_name

  }


  void readGroup(std::istringstream& is) {
    // g group_name1 group_name2 group_name3

    activeGroups_.clear();

    OBJGroup* previousGroup;

    if( objGroups_.size() == 0 ) {
      previousGroup = &dummyGroup_;
    } else {
      previousGroup = &objGroups_[objGroups_.size()-1];
    }

    bool previousGroupInfoExists = false;
    std::string name; 
    while( is >> std::ws >> name ) {
      OBJGroup newGroup;

      newGroup.name_ = name;
      newGroup.nVerticesSet_ = nVerticesSet_;
      newGroup.nUVsSet_ = nUVsSet_;
      newGroup.nNormalsSet_ = nNormalsSet_;
      newGroup.nVertices_ = nVertices_;
      newGroup.nUVs_ = nUVs_;
      newGroup.nNormals_ = nNormals_;

      OBJGroup otherGroupInstanceWithTheSameName;
      if( existsAnotherGroupInstance(name, otherGroupInstanceWithTheSameName) ) {
        previousGroupInfoExists = true;
        otherGroupInstanceWithTheSameName.copySpecificGroupDataTo(newGroup);
        if( !otherGroupInstanceWithTheSameName.generalGroupDataMatch(newGroup) ) {
          throw std::invalid_argument{ report_error("Instances of the same OBJGroup " << name << " has not matching general group data") };
        }
      }

      newGroup.globalOffsetVertices_ = previousGroup->globalOffsetVertices_ + previousGroup->globalSizeVertices_;
      newGroup.globalOffsetUVs_ = previousGroup->globalOffsetUVs_ +  previousGroup->globalSizeUVs_;
      newGroup.globalOffsetNormals_ = previousGroup->globalOffsetNormals_ + previousGroup->globalSizeNormals_;

      newGroup.globalSizeVertices_ = vertices_.size() - newGroup.globalOffsetVertices_;
      newGroup.globalSizeUVs_ = uvs_.size() - newGroup.globalOffsetUVs_;
      newGroup.globalSizeNormals_ = normals_.size() - newGroup.globalOffsetNormals_;

      objGroups_.push_back(newGroup);
      activeGroups_.push_back(&objGroups_[objGroups_.size()-1]);

    }   

    resetForNewGroup();

    if( previousGroupInfoExists ) {
      OBJGroup* anActiveGroup = activeGroups_[0];
      facesSet_ = anActiveGroup->facesSet_;
      facesHasVertex_ = anActiveGroup->facesHasVertex_;
      facesHasUv_ = anActiveGroup->facesHasUv_;
      facesHasNormal_ = anActiveGroup->facesHasNormal_;
      nVerticesSet_ = anActiveGroup->nVerticesSet_;
      nUVsSet_ = anActiveGroup->nUVsSet_;
      nNormalsSet_ = anActiveGroup->nNormalsSet_;
      nFacesGroupsSet_ = anActiveGroup->nFacesGroupsSet_;
      nVertices_ = anActiveGroup->nVertices_;
      nUVs_ = anActiveGroup->nUVs_;
      nNormals_ = anActiveGroup->nNormals_;
      nFacesGroups_ = anActiveGroup->nFacesGroups_;
    } 

  }


  void readComment(std::istringstream& is) {
    // # comment

  }


  void readMaterials(std::istringstream& is) {
    // mtllib filename1 filename2 . . .
    std::string file;
    while( is >> std::ws >> file ) {
      mtlParser_.parseFile(RecursiveParser<Parser>::getFilePath() + file);
    }
  }


  void useMaterial(std::istringstream& is) {
    // usemtl material_name
    std::string material;
    if(is >> std::ws >> material) {
      for(auto activeGroup : activeGroups_) {
        // if( activeGroup->materialSet_ == true && activeGroup->material_ != material ) {
        //   throw std::invalid_argument{ report_error("In OBJ file " 
        //                                              << RecursiveParser<Parser>::getFileName() << " @ " 
        //                                              << RecursiveParser<Parser>::getLineNumber()
        //                                              << " , the material for the group " << activeGroup->name_ 
        //                                              << " is already set, now it is " 
        //                                              << activeGroup->material_ << ", while the redefinition is trying to set it to "
        //                                              << material) };
        // }
        activeGroup->material_ = material;
        activeGroup->materialSet_ = true;
      }
    }

  }


  void smoothShading(std::istringstream& is) {
    // s group_number
    // s off      
    // s 0

  }


  void readValues(std::istringstream& is, std::vector<float>& vec, bool& nSet, unsigned int& n) {
    float value;
    unsigned int counter = 0;
    while( is >> std::ws >> value ) {
      vec.push_back(value);
      counter++;
    }

    if( nSet && counter != n ) {
      throw std::invalid_argument{ report_error("The OBJ file " 
                                                 << RecursiveParser<Parser>::getFileName() << " @ " 
                                                 << RecursiveParser<Parser>::getLineNumber()
                                                 << " is not coherent, "
                                                 << counter << " values read while it should have been "
                                                 << n << " values read") };
    } else {
      n = counter;
      nSet = true;
    }

  }


  void readFaces(std::istringstream& groupIs) {
    // f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ....
    // f v1 v2 v3 ....
    // f v1/vt1 v2/vt2 v3/vt3 ...
    // f v1//vn1 v2//vn2 v3//vn3 ...

    if( atTheFirstFace_ ) {
      for(auto activeGroup : activeGroups_) {
        activeGroup->globalOffsetVertexIndices_ = vertexIndices_.size();
        activeGroup->globalOffsetUVIndices_ = uvIndices_.size();
        activeGroup->globalOffsetNormalIndices_ = normalIndices_.size();
      }
      atTheFirstFace_ = false;
    }

    std::string group;
    unsigned int facesGroupCounter = 0;
    while( groupIs >> std::ws && std::getline(groupIs, group, ' ') ) {
      facesGroupCounter++;
      std::istringstream individualIs{group};

      unsigned short vertexIndex, uvIndex, normalIndex;
      bool hasVertexIndex = false, hasUvIndex = false, hasNormalIndex = false;

      if( individualIs >> std::ws >> vertexIndex ) {
        hasVertexIndex = true;
        if( individualIs.get() == '/' ) {
          if( individualIs.peek() == '/' ) {
            individualIs.get();
            if( individualIs >> std::ws >> normalIndex ) {
              hasNormalIndex = true;
            }
          } else {
            if( individualIs >> std::ws >> uvIndex ) {
              hasUvIndex = true;
              if( individualIs.get() == '/' ) {
                if( individualIs >> std::ws >> normalIndex ) {
                  hasNormalIndex = true;
                }
              }
            }
          }
        }    
      }


      if( facesSet_ ) {
        if( (facesHasVertex_ != hasVertexIndex) ||
            (facesHasUv_ != hasUvIndex) ||
            (facesHasNormal_ != hasNormalIndex) ) {

          throw std::invalid_argument{ report_error("The OBJ file " 
                                                    << RecursiveParser<Parser>::getFileName() << " @ " 
                                                    << RecursiveParser<Parser>::getLineNumber()
                                                    << " is not coherent, as a faces group is invalid:" << std::endl
                                                    << "facesSet_ = " << facesSet_ << std::endl
                                                    << "facesHasVertex_ = " << facesHasVertex_ << std::endl
                                                    << "facesHasUv_ = " << facesHasUv_ << std::endl
                                                    << "facesHasNormal_ = " << facesHasNormal_ << std::endl
                                                    << "hasVertexIndex = " << hasVertexIndex << std::endl
                                                    << "hasUvIndex = " << hasUvIndex << std::endl
                                                    << "hasNormalIndex = " << hasNormalIndex << std::endl
                                                    << "(facesHasVertex_ != hasVertexIndex) = " << (facesHasVertex_ != hasVertexIndex) << std::endl
                                                    << "(facesHasUv_ != hasUvIndex) = " << (facesHasUv_ != hasUvIndex) << std::endl
                                                    << "(facesHasNormal_ != hasNormalIndex) = " << (facesHasNormal_ != hasNormalIndex) ) };       
        }
      } else {

        for(auto activeGroup : activeGroups_) {
          activeGroup->facesHasVertex_ = hasVertexIndex;
          activeGroup->facesHasUv_ = hasUvIndex;
          activeGroup->facesHasNormal_ = hasNormalIndex;
          activeGroup->facesSet_ = true;
        }

        facesHasVertex_ = hasVertexIndex;
        facesHasUv_ = hasUvIndex;
        facesHasNormal_ = hasNormalIndex;
        facesSet_ = true;
      }

      if( hasVertexIndex ) {
        vertexIndices_.push_back(vertexIndicesReadOffset_ + vertexIndex-1);
      }

      if( hasUvIndex ) {
        uvIndices_.push_back(uvIndicesReadOffset_ + uvIndex-1);
      }

      if( hasNormalIndex ) {
        normalIndices_.push_back(normalIndicesReadOffset_ + normalIndex-1);
      }

    }


    if( nFacesGroupsSet_ ) {
      if( facesGroupCounter != nFacesGroups_ ) {
        throw std::invalid_argument{ report_error("The OBJ file " 
                                                  << RecursiveParser<Parser>::getFileName() << " @ " 
                                                  << RecursiveParser<Parser>::getLineNumber()
                                                  << " is not coherent, as the number of faces groups varies:" << std::endl
                                                  << "nFacesGroupsSet_ = " << nFacesGroupsSet_ << std::endl
                                                  << "nFacesGroups_ = " << nFacesGroups_ << std::endl
                                                  << "facesGroupCounter = " << facesGroupCounter << std::endl
                                                  << "(facesGroupCounter != nFacesGroups_) = " << (facesGroupCounter != nFacesGroups_) << std::endl ) };       
      }
    } else {
      for(auto activeGroup : activeGroups_) {
        activeGroup->nFacesGroups_ = facesGroupCounter;
        activeGroup->nFacesGroupsSet_ = true;
      }

      nFacesGroups_ = facesGroupCounter;
      nFacesGroupsSet_ = true;
    }

    for(auto activeGroup : activeGroups_) {
      activeGroup->globalSizeVertexIndices_ = vertexIndices_.size() -  activeGroup->globalOffsetVertexIndices_;
      activeGroup->globalSizeUVIndices_ = uvIndices_.size() -  activeGroup->globalOffsetUVIndices_;
      activeGroup->globalSizeNormalIndices_ = normalIndices_.size() -  activeGroup->globalOffsetNormalIndices_;
    }

  }


private:
  // All the groups read, including a dummy group at the first spot
  std::vector<OBJGroup> objGroups_;
  std::vector<OBJGroup*> activeGroups_;
  OBJGroup dummyGroup_;
  bool createdDummy_ = false;

  MTLParser<Parser> mtlParser_;

  // Flags for the group currently read
  bool atTheFirstFace_;
  bool facesSet_;
  bool facesHasVertex_;
  bool facesHasUv_;
  bool facesHasNormal_;

  // Flags for the data currently read
  bool nVerticesSet_;
  bool nUVsSet_;
  bool nNormalsSet_;
  bool nFacesGroupsSet_;
    
  // Temporary variables for the data currently read
  unsigned int nVertices_;
  unsigned int nUVs_;
  unsigned int nNormals_;
  unsigned int nFacesGroups_;

  // The global data DBs
  std::vector<float> vertices_;
  std::vector<float> uvs_;
  std::vector<float> normals_;

  // The global index DBs
  std::vector<unsigned short> vertexIndices_;
  std::vector<unsigned short> uvIndices_;
  std::vector<unsigned short> normalIndices_;

  // Offset that is used for the next parseFile invoke
  unsigned int vertexIndicesReadOffset_ = 0;
  unsigned int uvIndicesReadOffset_ = 0;
  unsigned int normalIndicesReadOffset_ = 0;


};


#endif // OBJPARSER_H