#ifndef OBJGROUP_H
#define OBJGROUP_H

struct OBJGroup {


  void copySpecificGroupDataTo(OBJGroup& objgrp) {
    objgrp.facesSet_ = facesSet_;
    objgrp.facesHasVertex_ = facesHasVertex_;
    objgrp.facesHasUv_ = facesHasUv_;
    objgrp.facesHasNormal_ = facesHasNormal_;
    objgrp.nFacesGroupsSet_ = nFacesGroupsSet_;
    objgrp.nFacesGroups_ = nFacesGroups_;

    objgrp.material_ = material_;
    objgrp.materialSet_ = materialSet_;
  }


  bool generalGroupDataMatch(OBJGroup& objgrp) {
    return (objgrp.facesSet_ == facesSet_) &&
           (objgrp.facesHasVertex_ == facesHasVertex_) &&
           (objgrp.facesHasUv_ == facesHasUv_) &&
           (objgrp.facesHasNormal_ == facesHasNormal_) &&
           (objgrp.nVerticesSet_ == nVerticesSet_) &&
           (objgrp.nUVsSet_ == nUVsSet_) &&
           (objgrp.nNormalsSet_ == nNormalsSet_) &&
           (objgrp.nFacesGroupsSet_ == nFacesGroupsSet_) &&
           (objgrp.nVertices_ == nVertices_ )&&
           (objgrp.nUVs_ == nUVs_) &&
           (objgrp.nNormals_ == nNormals_) &&
           (objgrp.nFacesGroups_ == nFacesGroups_) &&
           (objgrp.materialSet_ == materialSet_) &&
           (objgrp.material_ == material_);
  }


  void print() {
    std::cout << "------------------------" << std::endl;
    std::cout << "name_ = " << name_ << std::endl;
    std::cout << "material_ = " << material_ << std::endl;

    std::cout << "nVertices_ = " << nVertices_ << std::endl;
    std::cout << "nUVs_ = " << nUVs_ << std::endl;
    std::cout << "nNormals_ = " << nNormals_ << std::endl;
    std::cout << "nFacesGroups_ = " << nFacesGroups_ << std::endl;

    std::cout << "globalOffsetVertices_ = " << globalOffsetVertices_ << std::endl;
    std::cout << "globalSizeVertices_ = " << globalSizeVertices_ << std::endl;

    std::cout << "globalOffsetUVs_ = " << globalOffsetUVs_ << std::endl;
    std::cout << "globalSizeUVs_ = " << globalSizeUVs_ << std::endl;

    std::cout << "globalOffsetNormals_ = " << globalOffsetNormals_ << std::endl;
    std::cout << "globalSizeNormals_ = " << globalSizeNormals_ << std::endl;

    std::cout << "globalOffsetVertexIndices_ = " << globalOffsetVertexIndices_ << std::endl;
    std::cout << "globalSizeVertexIndices_ = " << globalSizeVertexIndices_ << std::endl;

    std::cout << "globalOffsetUVIndices_ = " << globalOffsetUVIndices_ << std::endl;
    std::cout << "globalSizeUVIndices_ = " << globalSizeUVIndices_ << std::endl;

    std::cout << "globalOffsetNormalIndices_ = " << globalOffsetNormalIndices_ << std::endl;
    std::cout << "globalSizeNormalIndices_ = " << globalSizeNormalIndices_ << std::endl;
  }


  std::string name_;
  std::string material_;

  bool materialSet_ = false;

  bool facesSet_ = false;
  bool facesHasVertex_ = false;
  bool facesHasUv_ = false;
  bool facesHasNormal_ = false;

  bool nVerticesSet_ = false;
  bool nUVsSet_ = false;
  bool nNormalsSet_ = false;
  bool nFacesGroupsSet_ = false;
  
  unsigned int nVertices_ = 0;
  unsigned int nUVs_ = 0;
  unsigned int nNormals_ = 0;
  unsigned int nFacesGroups_ = 0;


  unsigned int globalOffsetVertices_ = 0;
  unsigned int globalOffsetUVs_ = 0;
  unsigned int globalOffsetNormals_ = 0;

  unsigned int globalSizeVertices_ = 0;
  unsigned int globalSizeUVs_ = 0;
  unsigned int globalSizeNormals_ = 0;

  unsigned int globalOffsetVertexIndices_ = 0;
  unsigned int globalOffsetUVIndices_ = 0;
  unsigned int globalOffsetNormalIndices_ = 0;

  unsigned int globalSizeVertexIndices_ = 0;
  unsigned int globalSizeUVIndices_ = 0;
  unsigned int globalSizeNormalIndices_ = 0;

};

#endif // OBJGROUP_H