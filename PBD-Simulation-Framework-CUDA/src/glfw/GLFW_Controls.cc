#include "glfw/GLFW_Controls.h"

#define M_PI 3.14159265358979


void Scroll(GLFWwindow* window, double x, double y) {

  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);

  Events::scroll(xpos, ypos, x, y);

}


void Click(GLFWwindow* window, int button, int action, int mods) {

  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);

  Events::click(xpos, ypos, button, action, mods);

}


GLFW_Controls::GLFW_Controls(GLFWwindow* glfwWindow, 
              const unsigned int screenWidth, 
              const unsigned int screenHeight,
              const glm::vec3 defaultPosition,
              const glm::vec3 defaultDirection,
              const float defaultFov,
              const float nearPlaneDistance,
              const float farPlaneDistance,
              const float defaultMovementSpeed,
              const float defaultMouseSpeed) 
              : glfwWindow_(glfwWindow)
              , screenWidth_(screenWidth)
              , screenHeight_(screenHeight)
              , halfScreenWidth_(screenWidth/2)
              , halfScreenHeight_(screenHeight/2)
              , defaultPosition_(defaultPosition)
              , defaultDirection_(defaultDirection)
              , defaultFov_(defaultFov)
              , nearPlaneDistance_(nearPlaneDistance)
              , farPlaneDistance_(farPlaneDistance)
              , defaultMovementSpeed_(defaultMovementSpeed)
              , defaultMouseSpeed_(defaultMouseSpeed)
              , position_(defaultPosition)
              , direction_(defaultDirection)
              , fov_(defaultFov)
              , screenRatio_(static_cast<float>(screenWidth) / static_cast<float>(screenHeight))
              , movementSpeed_(defaultMovementSpeed)
              , mouseSpeed_(defaultMouseSpeed)
              , inGame_(true)
              , lastToggle_(0)
              , enteringWindow_(true)
              , reenteringWindow_(false) {

  double r = sqrt(pow(defaultDirection_.x, 2) + pow(defaultDirection_.y, 2)+ pow(defaultDirection_.z, 2));
  defaultHorizontalAngle_ = M_PI/2 - atan2(defaultDirection_.z, defaultDirection_.x);
  defaultVerticalAngle_ = M_PI/2 - acos(defaultDirection_.y / r);

  // std::cout << "R: " << r << std::endl;
  // std::cout << "H: " << (defaultHorizontalAngle_ * 180.0 / M_PI) << std::endl;
  // std::cout << "V: " << (defaultVerticalAngle_ * 180.0 / M_PI) << std::endl;

  setInputModeFreeLook();

  projectionMatrix_ = glm::perspective(fov_, screenRatio_, nearPlaneDistance_, farPlaneDistance_);

  handleInput();
  glfwSetCursorPos(glfwWindow_, halfScreenWidth_, halfScreenHeight_);

}

void GLFW_Controls::setInputModeFreeLook() {
  glfwSetInputMode(glfwWindow_, GLFW_STICKY_KEYS, GL_TRUE);
  glfwSetCursorPos(glfwWindow_, screenWidth_/2, screenHeight_/2);
  glfwSetInputMode(glfwWindow_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetScrollCallback(glfwWindow_, Scroll);
  glfwSetMouseButtonCallback(glfwWindow_, Click);
}


void GLFW_Controls::setInputModeStandard() {
  glfwSetInputMode(glfwWindow_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  glfwSetInputMode(glfwWindow_, GLFW_STICKY_KEYS, GL_TRUE);
  glfwSetScrollCallback(glfwWindow_, Scroll);
  glfwSetMouseButtonCallback(glfwWindow_, Click);
}


glm::vec3 GLFW_Controls::getCameraPosition() const {
  return position_;
}


glm::vec3 GLFW_Controls::getCameraDirection() const {
  return direction_;
}


glm::vec3 GLFW_Controls::getCameraRight() const {
  return right_;
}


glm::vec3 GLFW_Controls::getCameraUp() const {
  return up_;
}


glm::mat4 GLFW_Controls::getViewMatrix() const {
  return viewMatrix_;
}


glm::mat4 GLFW_Controls::getProjectionMatrix() const {
  return projectionMatrix_;
}

float GLFW_Controls::getCameraNear() const {
  return nearPlaneDistance_;
}

float GLFW_Controls::getCameraFar() const {
  return farPlaneDistance_;
}


void GLFW_Controls::computeCamera(double& xpos, double& ypos, float& deltaTime) {

  // Compute new orientation
  horizontalAngle_ += mouseSpeed_ * float(halfScreenWidth_ - xpos);
  verticalAngle_   += mouseSpeed_ * float(halfScreenHeight_ - ypos);

  // Direction : Spherical coordinates to Cartesian coordinates conversion
  direction_ = glm::vec3(
    cos(verticalAngle_) * sin(horizontalAngle_),
    sin(verticalAngle_),
    cos(verticalAngle_) * cos(horizontalAngle_)
  );

  right_ = glm::vec3(
    sin(horizontalAngle_ - M_PI/2.0f),
    0,
    cos(horizontalAngle_ - M_PI/2.0f)
  );

  // Head is up (set to 0,-1,0 to look upside-down)
  up_ = glm::cross(right_, direction_);

  // Move forward
  if( glfwGetKey(glfwWindow_, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(glfwWindow_, GLFW_KEY_UP) == GLFW_PRESS ){
    position_ += direction_ * deltaTime * movementSpeed_;
  }
  // Move backward
  if( glfwGetKey(glfwWindow_, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(glfwWindow_, GLFW_KEY_DOWN) == GLFW_PRESS ){
    position_ -= direction_ * deltaTime * movementSpeed_;
  }
  // Strafe right
  if( glfwGetKey(glfwWindow_, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(glfwWindow_, GLFW_KEY_RIGHT) == GLFW_PRESS ){
    position_ += right_ * deltaTime * movementSpeed_;
  }
  // Strafe left
  if( glfwGetKey(glfwWindow_, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(glfwWindow_, GLFW_KEY_LEFT) == GLFW_PRESS ){
    position_ -= right_ * deltaTime * movementSpeed_;
  }

}


bool GLFW_Controls::isInGame() const {
  return inGame_;
}



void GLFW_Controls::checkInGameToggle(double currentTime) {
  if( glfwGetKey(glfwWindow_, GLFW_KEY_M ) == GLFW_PRESS ) {

    if( currentTime - lastToggle_ > TOGGLE_COOL_DOWN ) { 
      lastToggle_ = currentTime;

      if( inGame_ ) {
        glfwSetInputMode(glfwWindow_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        glfwSetInputMode(glfwWindow_, GLFW_STICKY_KEYS, GL_TRUE);
      } else {
        glfwSetInputMode(glfwWindow_, GLFW_STICKY_KEYS, GL_FALSE);
        glfwSetInputMode(glfwWindow_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        reenteringWindow_ = true;
      }

      inGame_ = !inGame_;
    }

  }
}


void GLFW_Controls::handleInput() {

  double currentTime = glfwGetTime();

  checkInGameToggle(currentTime);

  if( !inGame_ ) {
    return;
  }

  static double lastTime = glfwGetTime();

  float deltaTime = float(currentTime - lastTime);

  double xpos, ypos;
  glfwGetCursorPos(glfwWindow_, &xpos, &ypos);

  if( enteringWindow_ ) {

    horizontalAngle_ = defaultHorizontalAngle_;
    verticalAngle_ = defaultVerticalAngle_;

    if( xpos == halfScreenWidth_ && ypos == halfScreenHeight_ ) {
      enteringWindow_ = false;
    }

    xpos = screenWidth_/2;
    ypos = screenHeight_/2;
  }

  if( reenteringWindow_ ) {
    if( xpos == halfScreenWidth_ && ypos == halfScreenHeight_ ) {
      reenteringWindow_ = false;
    }
    xpos = screenWidth_/2;
    ypos = screenHeight_/2;
  }

  // Reset mouse position for next frame
  glfwSetCursorPos(glfwWindow_, halfScreenWidth_, halfScreenHeight_);

  computeCamera(xpos, ypos, deltaTime);

  viewMatrix_ = glm::lookAt(position_, position_ + direction_, up_);

  // For the next frame, the "last time" will be "now"
  lastTime = currentTime;

}