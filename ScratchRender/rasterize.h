#pragma once

#include <SDL.h>
#include "Scene.h"
#include "Context.h"

void rasterize(SDL_Renderer* renderer, Scene* scene, Context context);