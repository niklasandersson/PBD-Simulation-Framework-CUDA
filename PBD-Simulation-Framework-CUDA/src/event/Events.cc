#include "Events.h"


// Direct_Call_Event<void()> Events::close_console;


// Global_Defer_Call_Event<void(const bool is_running)> Events::engine_set_is_running;
// Global_Defer_Call_Event<void(const bool is_running)> Events::engine_set_is_simulating;
Global_Defer_Call_Event<void(const double position_x, const double position_y, const int button, const int action, const int mods)> Events::click;
Global_Defer_Call_Event<void(const double position_x, const double position_y, const double offset_x, const double offset_y)> Events::scroll;

Global_Defer_Call_Event<void(glm::vec3 pos, glm::vec3 dir)> Events::addParticle;

//Local_Defer_Call_Event<void(std::vector<glm::vec3> vertices, std::vector<unsigned short> indices, std::vector<glm::vec3> colors, std::vector<unsigned int> tilePath)> Events::update;

//Local_Defer_Call_Event<void(const glm::vec3 pos, const bool add)> Events::cityChange;


// Fluid
Direct_Call_Event<void()> Events::stopEngine;


