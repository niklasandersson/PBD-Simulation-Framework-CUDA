#ifndef EVENTS_H
#define EVENTS_H

#include "glm/glm.hpp"

#include "event/Delegate.h"
#include "event/Global_Event_Base.h"

#include "event/Direct_Call_Event.h"
#include "event/Local_Defer_Call_Event.h"
#include "event/Global_Defer_Call_Event.h"


struct Events {

  // Direct Call Events:

  // Local Defer Call Events:

  // Global Defer Call Events:
  // static Direct_Call_Event<void()> close_console;

  // static Global_Defer_Call_Event<void(const bool is_running)> engine_set_is_running;
  // static Global_Defer_Call_Event<void(const bool is_running)> engine_set_is_simulating;
  static Global_Defer_Call_Event<void(const double position_x, const double position_y, const int button, const int action, const int mods)> click;
  static Global_Defer_Call_Event<void(const double position_x, const double position_y, const double offset_x, const double offset_y)> scroll;

  // Local_Defer_Call_Event<void(std::vector<glm::vec3> vertices, std::vector<unsigned short> indices, std::vector<glm::vec3> colors, std::vector<unsigned int> tilePath)> update;

  //static Local_Defer_Call_Event<void(const glm::vec3 pos, const bool add)> cityChange;



  // Fluid
  static Direct_Call_Event<void()> stopEngine;



};


#endif // EVENTS_H