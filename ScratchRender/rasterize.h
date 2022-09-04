#pragma once

#include <SDL.h>
#include "Scene.h"

void rasterize(SDL_Renderer* renderer, Scene* scene, int width, int height);