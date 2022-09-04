#include "mathutils.h"
#include "mathutils_kernel.cuh"

#include <iostream>
#include <cassert>

Vector* rotateX(Vector point, float angle)
{
	float* rotX = (float*)malloc(3 * 3 * sizeof(float));
	rotX[0] = 1;  rotX[1] = 0;           rotX[2] = 0;
	rotX[3] = 0;  rotX[4] = cos(angle);  rotX[5] = -sin(angle);
	rotX[6] = 0;  rotX[7] = sin(angle); rotX[8] = cos(angle);

	float* xyz = (float*)malloc(3 * sizeof(float));
	xyz[0] = point.x;
	xyz[1] = point.y;
	xyz[2] = point.z;

	mult_mv(rotX, xyz, xyz, 3);

	Vector* res = new Vector(xyz[0], xyz[1], xyz[2]);

	free(rotX);
	free(xyz);

	return res;
}

Vector* rotateY(Vector point, float angle)
{
	float* rotY = (float*)malloc(3 * 3 * sizeof(float));
	rotY[0] = cos(angle);  rotY[1] = 0;  rotY[2] = sin(angle);
	rotY[3] = 0;           rotY[4] = 1;  rotY[5] = 0;
	rotY[6] = -sin(angle); rotY[7] = 0;  rotY[8] = cos(angle);

	float* xyz = (float*)malloc(3 * sizeof(float));
	xyz[0] = point.x;
	xyz[1] = point.y;
	xyz[2] = point.z;

	mult_mv(rotY, xyz, xyz, 3);

	Vector* res = new Vector(xyz[0], xyz[1], xyz[2]);

	free(rotY);
	free(xyz);

	return res;
}

Vector* rotateZ(Vector point, float angle)
{
	float* rotZ = (float*)malloc(3 * 3 * sizeof(float));
	rotZ[0] = cos(angle); rotZ[1] = -sin(angle); rotZ[2] = 0;
	rotZ[3] = sin(angle); rotZ[4] = cos(angle);  rotZ[5] = 0;
	rotZ[6] = 0;          rotZ[7] = 0;           rotZ[8] = 1;

	float* xyz = (float*)malloc(3 * sizeof(float));
	xyz[0] = point.x;
	xyz[1] = point.y;
	xyz[2] = point.z;

	mult_mv(rotZ, xyz, xyz, 3);

	Vector* res = new Vector(xyz[0], xyz[1], xyz[2]);

	free(rotZ);
	free(xyz);

	return res;
}

Vector* rotateAxis(Vector point, Vector axis, float angle)
{
	
	Vector norm = axis.normalized();
	float ux = norm.x;
	float uy = norm.y;
	float uz = norm.z;


	float* rot = (float*)malloc(3 * 3 * sizeof(float));
	rot[0] = cos(angle) + ux * ux * (1-cos(angle));         rot[1] = ux * uy * (1 - cos(angle)) - uz * sin(angle); rot[2] = ux * uz * (1 - cos(angle)) + uy * sin(angle);
	rot[3] = uy * ux * (1 - cos(angle)) + uz * sin(angle);  rot[4] = cos(angle) + uy * uy * (1-cos(angle));        rot[5] = uy * uz * (1 - cos(angle)) - ux * sin(angle);
	rot[6] = uz * ux * (1 - cos(angle)) - uy * sin(angle);  rot[7] = uz * uy * (1 - cos(angle)) + ux * sin(angle); rot[8] = cos(angle) + uz * uz * (1 - cos(angle));

	float* xyz = (float*)malloc(3 * sizeof(float));
	xyz[0] = point.x;
	xyz[1] = point.y;
	xyz[2] = point.z;

	mult_mv(rot, xyz, xyz, 3);
	
	Vector* res = new Vector(xyz[0], xyz[1], xyz[2]);
	free(rot);
	free(xyz);

	return res;
}

Vector* scale(Vector point, Vector factor)
{
	float* scaleMat = (float*)malloc(3 * 3 * sizeof(float));
	scaleMat[0] = factor.x; scaleMat[1] = 0;       scaleMat[2] = 0;       
	scaleMat[3] = 0;       scaleMat[4] = factor.y; scaleMat[5] = 0;
	scaleMat[6] = 0;       scaleMat[7] = 0;       scaleMat[8] = factor.z;

	float* xyz = (float*)malloc(3 * sizeof(float));
	xyz[0] = point.x;
	xyz[1] = point.y;
	xyz[2] = point.z;

	mult_mv(scaleMat, xyz, xyz, 3);

	Vector* res = new Vector(xyz[0], xyz[1], xyz[2]);

	free(scaleMat);
	free(xyz);

	return res;
}

Vector* translate(Vector point, Vector location)
{

	float* xyz = (float*)malloc(4 * sizeof(float));
	xyz[0] = point.x;
	xyz[1] = point.y;
	xyz[2] = point.z;
	xyz[3] = 1;

	float* translate = (float*)malloc(4 * 4 * sizeof(float));

	translate[0] = 1;  translate[1] = 0;  translate[2] = 0;  translate[3] = location.x;
	translate[4] = 0;  translate[5] = 1;  translate[6] = 0;  translate[7] = location.y;
	translate[8] = 0;  translate[9] = 0;  translate[10] = 1; translate[11] = location.z;
	translate[12] = 0; translate[13] = 0; translate[14] = 0; translate[15] = 1;

	mult_mv(translate, xyz, xyz, 4);

	Vector* res = new Vector(xyz[0], xyz[1], xyz[2]);

	free(translate);
	free(xyz);

	return res;

}

// *********************************************************************************************

Vector* transform_point(Vector location, Euler rotation, Vector scale_factor, Vector point)
{
	Vector* pos = rotateX(point, rotation.x);
	pos = rotateY(*pos, rotation.y);
	pos = rotateZ(*pos, rotation.z);
	pos = translate(*pos, location);
	pos = scale(*pos, scale_factor);
	
	return pos;
}

// *********************************************************************************************

float remap(float s, float a1, float a2, float b1, float b2)
{
	assert(a1 < a2);
	assert(b1 < b2);

	return (b1 + (s - a1) * (b2 - b1) / (a2 - a1));
}
