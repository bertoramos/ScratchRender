// CUDA_SDL_Test.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

// https://lazyfoo.net/tutorials/SDL/01_hello_SDL/windows/msvc2019/index.php
// https://medium.com/@aviatorx/c-and-cuda-project-visual-studio-d07c6ad771e3

#include <iostream>
#include <SDL.h>

#include <ctime>
#include <cstdlib>
#include <boost/algorithm/string.hpp>

#include "Vector.h"
#include "Model.h"
#include "Scene.h"

#include "rasterize.h"

const int SCREEN_WIDTH = 600;
const int SCREEN_HEIGHT = 600;

Scene* scene;
Model* model;
Camera* camera;

void setup(SDL_Renderer* renderer) {
	scene = new Scene;

	model = new Model("cube.obj");
	model->location = Vector(0, 0, 0);
	model->rotation = Euler(0, 0, 0);
	model->scale = Vector(50,50,50);


	camera = new Camera;
	camera->zNear = 10;
	camera->zFar = 2000;
	camera->fov = 3.14159286 / 3;
	camera->dolly(-50);
	camera->pan(3.1415 / 4);

	std::cout << camera->center << "\n";
	std::cout << camera->forward << "\n";
	std::cout << camera->up << "\n";
	//camera->near = 40;
	//camera->
	//camera->center = Vector(0,0,10);

	//camera->boom(-40);
	//camera->tilt(-45);

	scene->models.push_back(model);
	scene->camera = camera;
}

void draw(SDL_Renderer* renderer) {
	rasterize(renderer, scene, SCREEN_WIDTH, SCREEN_HEIGHT);

	SDL_RenderPresent(renderer);
}

void run_event(SDL_Event event) {

}

// -----------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	
    // SDL TEST
	srand(time(0));

	//The window we'll be rendering to
	SDL_Window* window = NULL;
	SDL_Renderer* renderer = NULL;

	//The surface contained by the window
	SDL_Surface* screenSurface = NULL;

	//Initialize SDL
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
	}
	else
	{
		//Create window
		SDL_CreateWindowAndRenderer(SCREEN_WIDTH, SCREEN_HEIGHT, 0, &window, &renderer);
		if (window == NULL)
		{
			printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
		}
		else
		{
			SDL_bool done = SDL_FALSE;

			setup(renderer);

			while (!done) {
				SDL_Event event;

				SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
				SDL_RenderClear(renderer);

				draw(renderer);

				while (SDL_PollEvent(&event)) {
					if (event.type == SDL_QUIT) {
						done = SDL_TRUE;
					}

					run_event(event);
				}
			}
		}
	}

	//Destroy window
	SDL_DestroyWindow(window);

	//Quit SDL subsystems
	SDL_Quit();

	return 0;
}
