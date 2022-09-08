// CUDA_SDL_Test.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

// https://lazyfoo.net/tutorials/SDL/01_hello_SDL/windows/msvc2019/index.php
// https://medium.com/@aviatorx/c-and-cuda-project-visual-studio-d07c6ad771e3

#include <iostream>
#include <SDL.h>

#include <ctime>
#include <cstdlib>
#include <boost/algorithm/string.hpp>

#include <cuda_runtime.h>

#include <ctime>

#include "Vector.h"
#include "Model.h"
#include "Scene.h"

#include "rasterize.h"
#include "Context.h"

Context context(600, 600);

Scene* scene;
Model* model;
Camera* camera;

int fps = 0;
int FRAMES = 60;
int frames = 60;
long long start = 0.;

void setup(SDL_Renderer* renderer) {
	scene = new Scene;

	model = new Model("cube.obj");
	model->location = Vector(0, 0, 0);
	model->rotation = Euler(0, 0, 0);
	model->scale = Vector(50,50,50);


	camera = new Camera;
	camera->zNear = 10;
	camera->zFar = 200;
	camera->fov = 3.14159286 / 3;
	camera->dolly(-100);
	camera->truck(-60);
	camera->pan(3.1415 / 4);
	camera->tilt(3.1415 / 4);
	camera->boom(40);
	camera->dolly(-100);

	std::cout << camera->center << "\n";
	std::cout << camera->forward << "\n";
	std::cout << camera->up << "\n";

	scene->models.push_back(model);
	scene->camera = camera;

}

void draw(SDL_Renderer* renderer) {
	rasterize(renderer, scene, context);

	SDL_RenderPresent(renderer);
}

void run_event(SDL_Event event) {
	
}

// -----------------------------------------------------------------------------------------------

void matmul_example();
void mv_mul_example();
void invert_test();

int main(int argc, char* argv[])
{
	
    // SDL TEST
	srand(time(0));

	//The window we'll be rendering to
	SDL_Window* window = NULL;
	SDL_Renderer* renderer = NULL;

	//The surface contained by the window
	SDL_Surface* screenSurface = NULL;

	float start_time, fps, frame_time;

	const int fps_times = 10;
	int fps_times_counter = fps_times;

	//Initialize SDL
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
	}
	else
	{
		//Create window
		SDL_CreateWindowAndRenderer(context.width, context.height, 0, &window, &renderer);
		if (window == NULL)
		{
			printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
		}
		else
		{
			SDL_bool done = SDL_FALSE;

			setup(renderer);

			while (!done) {
				start_time = SDL_GetTicks();

				SDL_Event event;

				SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
				SDL_RenderClear(renderer);

				draw(renderer);

				frame_time = SDL_GetTicks() - start_time;
				fps = (frame_time > 0) ? 1000.0f / frame_time : 0.0f;

				fps_times_counter--;
				if (fps_times_counter == 0) {
					std::cout << "FPS: " << fps << std::endl;
					fps_times_counter = fps_times;
				}

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

	/**************/
	//invert_test();

	cudaDeviceReset();

	return 0;
}

// ****************

#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_api.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mathutils_kernel.cuh"


void invert_test() {
	float c_values[9];
	float a_values[] = {
	  2,   3,   1,
	  1,   2,   1,
	  0,   0,   1 };

	float** a = (float**)malloc(sizeof(float*) * 1);
	float** c = (float**)malloc(sizeof(float*) * 1);
	a[0] = a_values;
	c[0] = c_values;
	

	invert(a, c, 3, 1);

	free(a);
	free(c);

	//for (size_t i = 0; i < 9; i++)
	//{
	//	c_values[i] = (*c)[i];
	//}
	for (size_t i = 0; i < 9; i++)
	{
		std::cout << c_values[i] << ",";
	}
	std::cout << "\n";
}

void mv_mul_example() {
	cudaError_t error;
	cublasStatus_t state;
	cublasHandle_t handle;


	int dim = 3;

	float* A = (float*)malloc(sizeof(float) * dim * dim);
	float* B = (float*)malloc(sizeof(float) * dim);
	float* C = (float*)malloc(sizeof(float) * dim * dim);

	A[0] = 2.; A[1] = 2.; A[2] = 2.;
	A[3] = 9.; A[4] = 6.; A[5] = 1.;
	A[6] = 8.; A[7] = 7.; A[8] = 3.;

	B[0] = 4.; B[1] = 1.; B[2] = 1.;

	C[0] = 0.; C[1] = 0.; C[2] = 0.;

	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, dim * dim * sizeof(float));
	cudaMalloc(&d_B, dim * sizeof(float));
	cudaMalloc(&d_C, dim * sizeof(float));

	error = cudaMemcpy(d_A, A, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "Error while copying data to device\n";
	}
	error = cudaMemcpy(d_B, B, dim * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "Error while copying data to device\n";
	}

	// ***
	int m = dim, n = dim;
	int lda = dim;
	const float alf = 1;
	const float bet = 0;
	const float* alpha = &alf;
	const float* beta = &bet;


	state = cublasCreate(&handle);
	if (state != CUBLAS_STATUS_SUCCESS) {
		std::cout << "Error in cublasCreate\n";
	}

	state = cublasSgemv(handle, CUBLAS_OP_T, m, n, alpha, d_A, lda, d_B, 1, beta, d_C, 1);
	if (state != CUBLAS_STATUS_SUCCESS) {
		std::cout << "Error in sgemv\n";
	}

	error = cudaMemcpy(C, d_C, dim * sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "Error while copying data to host " << error << "\n";
	}


	// ***

	std::cout << "C:\n";
	for (int i = 0; i < dim; i++)
	{
		std::cout << C[i] << " ";
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cublasDestroy(handle);

	free(A);
	free(B);
	free(C);

	cudaDeviceReset();
}

void matmul_example() {
	// (A*B).T = B.T * A.T

	cudaError_t error;
	cublasStatus_t state;
	cublasHandle_t handle;


	int dim = 3;

	float* A = (float*)malloc(sizeof(float) * dim * dim);
	float* B = (float*)malloc(sizeof(float) * dim * dim);
	float* C = (float*)malloc(sizeof(float) * dim * dim);

	A[0] = 2.; A[1] = 2.; A[2] = 2.;
	A[3] = 9.; A[4] = 6.; A[5] = 1.;
	A[6] = 8.; A[7] = 7.; A[8] = 3.;

	B[0] = 4.; B[1] = 1.; B[2] = 1.;
	B[3] = 3.; B[4] = 2.; B[5] = 6.;
	B[6] = 1.; B[7] = 0.; B[8] = 3.;

	C[0] = 0.; C[1] = 0.; C[2] = 0.;
	C[3] = 0.; C[4] = 0.; C[5] = 0.;
	C[6] = 0.; C[7] = 0.; C[8] = 0.;

	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, dim * dim * sizeof(float));
	cudaMalloc(&d_B, dim * dim * sizeof(float));
	cudaMalloc(&d_C, dim * dim * sizeof(float));

	error = cudaMemcpy(d_A, A, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "Error while copying data to device\n";
	}
	error = cudaMemcpy(d_B, B, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "Error while copying data to device\n";
	}

	// ***
	int m = dim, n = dim, k = dim;
	int lda = dim, ldb = dim, ldc = dim;
	const float alf = 1;
	const float bet = 0;
	const float* alpha = &alf;
	const float* beta = &bet;


	state = cublasCreate(&handle);
	if (state != CUBLAS_STATUS_SUCCESS) {
		std::cout << "Error in cublasCreate\n";
	}

	state = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_B, ldb, d_A, lda, beta, d_C, ldc);
	if (state != CUBLAS_STATUS_SUCCESS) {
		std::cout << "Error in sgemm\n";
	}

	error = cudaMemcpy(C, d_C, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "Error while copying data to host\n";
	}


	// ***

	std::cout << "C:\n";
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++) {
			std::cout << C[i * dim + j] << " ";
		}
		std::cout << "\n";
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cublasDestroy(handle);

	free(A);
	free(B);
	free(C);

	cudaDeviceReset();
}
