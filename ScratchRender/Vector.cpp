
#include "Vector.h"
#include <iostream>

Vector& Vector::operator=(const Vector& vector)
{
	x = vector.x;
	y = vector.y;
	z = vector.z;
	w = vector.w;

	return *this;
}

Vector& Vector::operator+(const Vector& vector)
{
	Vector* res = new Vector();
	res->x = x + vector.x;
	res->y = y + vector.y;
	res->z = z + vector.z;
	res->w = w + vector.w;
	return *res;
}

Vector& Vector::operator-(const Vector& vector)
{
	Vector* res = new Vector();
	res->x = x - vector.x;
	res->y = y - vector.y;
	res->z = z - vector.z;
	res->w = w - vector.w;
	return *res;
}

Vector& Vector::operator*(const Vector& vector)
{
	Vector* res = new Vector();
	res->x *= vector.x;
	res->y *= vector.y;
	res->z *= vector.z;
	res->w *= vector.w;
	return *res;
}

float Vector::operator&(const Vector& vector)
{
	float res = 0;
	res += x * vector.x;
	res += y * vector.y;
	res += z * vector.z;
	res += w * vector.w;
	return res;
}

void Vector::normalize() {
	float len = this->length();
	x /= len;
	y /= len;
	z /= len;
	w /= len;
}

Vector& Vector::normalized() {
	Vector* vec = new Vector(*this);
	vec->normalize();
	return *vec;
}

Vector& Vector::cross(const Vector& vector)
{
	float a1 = x;
	float a2 = y;
	float a3 = z;

	float b1 = vector.x;
	float b2 = vector.y;
	float b3 = vector.z;

	float i = a2 * b3 - a3 * b2;
	float j = a3 * b1 - a1 * b3;
	float k = a1 * b2 - a2 * b1;

	return *(new Vector(i, j, k));
}
