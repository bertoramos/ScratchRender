#pragma once
#include "Euler.h"
#include "Vector.h"

Vector* rotateX(Vector point, float angle);
Vector* rotateY(Vector point, float angle);
Vector* rotateZ(Vector point, float angle);
Vector* rotateAxis(Vector point, Vector axis, float angle);

Vector* scale(Vector point, Vector factor);
Vector* translate(Vector point, Vector location);

Vector* transform_point(Vector location, Euler rotation, Vector scale, Vector point);

float remap(float s, float a1, float a2, float b1, float b2);