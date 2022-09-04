#pragma once

#include <vector>

#include "Model.h"
#include "Camera.h"

class Scene
{
public:
	std::vector<Model*> models;
	Camera* camera;

};

