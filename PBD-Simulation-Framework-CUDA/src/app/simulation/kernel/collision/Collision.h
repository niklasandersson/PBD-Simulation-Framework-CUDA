#ifndef SORT_H
#define SORT_H

#include <limits>

#include "app/simulation/Parameters.h"

#include "Hash.h"

void initializeCollision(Parameters* parameters);
void solveCollisions(Parameters* parameters);

#endif // SORT_H