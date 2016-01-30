#ifndef EVENTS_H
#define EVENTS_H

#include "glm/glm.hpp"

#include "event/Delegate.h"
#include "event/Global_Event_Base.h"

#include "event/Direct_Call_Event.h"
#include "event/Local_Defer_Call_Event.h"
#include "event/Global_Defer_Call_Event.h"


struct Events {
  static Global_Defer_Call_Event<void(const double position_x, const double position_y, const int button, const int action, const int mods)> click;
  static Global_Defer_Call_Event<void(const double position_x, const double position_y, const double offset_x, const double offset_y)> scroll;

  static Local_Defer_Call_Event<void(glm::vec3 pos, glm::vec3 dir)> addParticle;
  static Local_Defer_Call_Event<void(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col)> addParticles;
  static Local_Defer_Call_Event<void()> clearParticles;
  static Local_Defer_Call_Event<void()> reload;
  static Local_Defer_Call_Event<void(const std::string file)> load;

  static Direct_Call_Event<void()> stopEngine;
};


#endif // EVENTS_H