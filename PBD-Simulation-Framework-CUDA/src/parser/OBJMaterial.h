#ifndef OBJMATERIAL_H
#define OBJMATERIAL_H

#include <iostream>
#include <glm/glm.hpp>


struct OBJMaterial {


  OBJMaterial(const std::string& name) : name_{name} { 
    // std::cout << "New material: " << name << std::endl;
  }


  void print() {
    std::cout << "----------------" << std::endl;
    std::cout << "name_: " << name_ << std::endl;
    std::cout << "ka_: " << ka_.x << " " << ka_.y << " " << ka_.z << std::endl;
    std::cout << "kd_: " << kd_.x << " " << kd_.y << " " << kd_.z << std::endl;
    std::cout << "ks_: " << ks_.x << " " << ks_.y << " " << ks_.z << std::endl;
    std::cout << "ke_: " << ke_.x << " " << ke_.y << " " << ke_.z << std::endl;
  }


  std::string name_;

  glm::vec3 ka_, kd_, ks_, ke_;

};

#endif // OBJMATERIAL_H