#include "Events.h"


Global_Defer_Call_Event<void(const double position_x, const double position_y, const int button, const int action, const int mods)> Events::click;
Global_Defer_Call_Event<void(const double position_x, const double position_y, const double offset_x, const double offset_y)> Events::scroll;

Local_Defer_Call_Event<void(glm::vec3 pos, glm::vec3 dir)> Events::addParticle;
Local_Defer_Call_Event<void(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col)> Events::addParticles;
Local_Defer_Call_Event<void()> Events::clearParticles;
Local_Defer_Call_Event<void()> Events::reload;

Direct_Call_Event<void()> Events::stopEngine;
