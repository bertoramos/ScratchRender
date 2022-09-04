#pragma once
class Euler
{
public:
	float x;
	float y;
	float z;

	Euler() : x(0), y(0), z(0) {}
	
	Euler(float x, float y, float z) : x(x), y(y), z(z) {}

	// Copy constructor
	Euler(Euler& euler) {
		x = euler.x;
		y = euler.y;
		z = euler.z;
	}

	Euler& operator= (const Euler& euler);
};
