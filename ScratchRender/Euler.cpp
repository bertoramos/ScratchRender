#include "Euler.h"

Euler& Euler::operator=(const Euler& euler)
{
	x = euler.x;
	y = euler.y;
	z = euler.z;

	return *this;
}
