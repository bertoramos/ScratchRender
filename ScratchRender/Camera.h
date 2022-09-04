#pragma once

#include "Vector.h"
#include "Euler.h"
#include "mathutils.h"

class Camera
{
public:
	Vector center;
	Vector forward;
	Vector up;
	float fov;
	float aspect;
	float zNear;
	float zFar;
	float top;
	float right;

	Camera() {
		forward = Vector(0, 0, 1);
		center = Vector(0, 0, 1);
		up = Vector(0,1,0);
		fov = 2.14159286 / 2; // 60º
		zNear = 20;
		zFar  = 80;
	}

	Vector getRight() {
		return up.cross(forward);
	}

	void tilt(float angle)
	{
		// Over right axis
		Vector R = getRight();

		forward = *rotateAxis(forward, R, angle);
		up = *rotateAxis(up, R, angle);
		forward.normalize();
		up.normalize();
	}

	void pan(float angle)
	{
		// Over up axis
		forward = *rotateAxis(forward, up, angle);
		forward.normalize();
	}

	void roll(float angle)
	{
		// Over forward axis
		up = *rotateAxis(up, forward, angle);
		up.normalize();
	}

	void dolly(float p)
	{
		// Translate in forward axis
		center.x += p * forward.x;
		center.y += p * forward.y;
		center.z += p * forward.z;
	}

	void boom(float p)
	{
		// Translate in up axis
		center.x += p * up.x;
		center.y += p * up.y;
		center.z += p * up.z;
	}

	void truck(float p)
	{
		// Translate in right axis
		Vector right = getRight();
		center.x += p * right.x;
		center.y += p * right.y;
		center.z += p * right.z;
	}
};
