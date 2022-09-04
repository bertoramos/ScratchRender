#pragma once

#include "./Vector.h"
#include "./Euler.h"

#include <string>
#include <vector>

class Model {

public:
	Vector location;
	Euler rotation;
	Vector scale;
	
	// Geometry
	std::vector<Vector*> * vertices;
	std::vector<Vector*> * normals;
	std::vector<std::vector<int>> *faces_vertex; // faces_vertex[i] = [0, 1, 2] // Indices en "vertices" para cada vertex en la face
	std::vector<std::vector<int>> *faces_normal; // faces_normal[i] = [0, 0, 0] // Indices en normals para cada vertex en la face

	// Materials

	Model(std::string obj);

};

void parse(std::string line, Model* model);
