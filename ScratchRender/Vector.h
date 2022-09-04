#pragma once

#include <cmath>
#include <ostream>

class Vector
{
public:
	float x;
	float y;
	float z;
	float w;

	Vector() : x(0), y(0), z(0), w(1) {}

	Vector(float x, float y, float z): x(x), y(y), z(z), w(1) {}

	Vector(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

	// Copy constructor
	Vector(const Vector& vec) {
		x = vec.x;
		y = vec.y;
		z = vec.z;
		w = vec.w;
	}

	Vector& operator= (const Vector& vector);

	Vector& operator+(const Vector& vector);
	Vector& operator-(const Vector& vector);
	Vector& operator*(const Vector& vector);
	float operator&(const Vector& vector);

	friend std::ostream& operator<<(std::ostream& os, const Vector& vec) {
		return os << "Vector(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
	}

	Vector& cross(const Vector& vector);

	void normalize();
	Vector& normalized();

	float length() {
		return sqrt(x * x + y * y + z * z);
	}

	
};
