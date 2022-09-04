
#include <iostream>

#include "rasterize.h"
#include "mathutils.h"
#include "mathutils_kernel.cuh"

void rasterize(SDL_Renderer* renderer, Scene* scene, int width, int height)
{
	Camera* cam = scene->camera;

	for (int m = 0; m < scene->models.size(); m++) {
		
		Model* model = scene->models.at(m);

		// Transform points
		std::vector<Vector*> verts;
		for (int v = 0; v < model->vertices->size(); v++) {
			Vector* pt = model->vertices->at(v);

			pt = transform_point(model->location,
						 		 model->rotation,
								 model->scale,
								 *pt);
			
			verts.push_back(pt);
		}

		// Proyect points
		float near  = cam->zNear;
		float far   = cam->zFar;
		float top   = cam->top;
		float right = cam->right;

		Vector cam_right = cam->getRight();

		// Transformation matrices
		// FIXME : INVERT MATRIX
		float* view_transform = (float*)malloc(4 * 4 * sizeof(float));

		view_transform[0] = cam_right.x; view_transform[1] = cam->up.x; view_transform[2] = cam->forward.x; view_transform[3] = cam->center.x;
		view_transform[4] = cam_right.y; view_transform[5] = cam->up.y; view_transform[6] = cam->forward.y; view_transform[7] = cam->center.y;
		view_transform[8] = cam_right.z; view_transform[9] = cam->up.z; view_transform[10] = cam->forward.z; view_transform[11] = cam->center.z;
		view_transform[12] = 0;        view_transform[13] = 0;          view_transform[14] = 0;              view_transform[15] = 1;

		float* perspective = (float*)malloc(4 * 4 * sizeof(float));
		float aspect = width / height;
		float tan_fov = tan(cam->fov / 2);

		perspective[ 0] = 1 / (aspect * tan_fov); perspective[ 1] = 0;          perspective[ 2] = 0;                            perspective[3] = 0;
		perspective[ 4] = 0;                      perspective[ 5] = 1/tan_fov;  perspective[ 6] = 0;                            perspective[7] = 0;
		perspective[ 8] = 0;                      perspective[ 9] = 0;          perspective[10] = (-far - near) / (far - near); perspective[11] = (-2 * far * near) / (far - near);
		perspective[12] = 0;                      perspective[13] = 0;          perspective[14] = -1;                           perspective[15] = 0;

		float* xyz = (float*)malloc(4 * sizeof(float));

		std::vector<Vector*> proyected;
		for (int v = 0; v < verts.size(); v++) {
			Vector* pt = verts.at(v);

			float world_x = pt->x;
			float world_y = pt->y;
			float world_z = pt->z;
			
			xyz[0] = world_x;
			xyz[1] = world_y;
			xyz[2] = world_z;
			xyz[3] = 1;
			
			// World space -> Camera space
			mult_mv(view_transform, xyz, xyz, 4);

			// Camera space -> NDC
			mult_mv(perspective, xyz, xyz, 4);

			//std::cout << xyz[0] << "," << xyz[1] << "," << xyz[2] << "," << xyz[3] << std::endl;
			//std::cout << xyz[0]/ xyz[3] << "," << xyz[1]/ xyz[3] << "," << xyz[2] / xyz[3] << std::endl;

			float ndc_x = xyz[0] / xyz[3];
			float ndc_y = xyz[1] / xyz[3];
			float ndc_z = xyz[2] / xyz[3];

			// NDC -> Screen space
			float screen_x = remap(ndc_x, -1, 1, 0, width);
			float screen_y = remap(ndc_y, -1, 1, 0, height);
			float depth    = remap(ndc_z, -1, 1, near, far);

			proyected.push_back(new Vector(screen_x, screen_y, depth));

			//std::cout << screen_x << " " << screen_y << "\n";

			//SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
			//SDL_RenderDrawPoint(renderer, screen_x, screen_y);
		}
		//std::cout << "out\n";

		free(xyz);
		free(perspective);
		free(view_transform);

		// Draw edges
		for (int f = 0; f < model->faces_vertex->size(); f++)
		{
			std::vector<int> face = model->faces_vertex->at(f);
			
			for (int v = 1; v < face.size(); v++)
			{ // 0-1 1-2 
				Vector *a = proyected.at(face.at(v - 1) - 1);
				Vector *b = proyected.at(face.at(v) - 1);

				SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
				SDL_RenderDrawLine(renderer, a->x, a->y, b->x, b->y);
			}
			// 2-0
			Vector *a = proyected.at(face.at(0) - 1);
			Vector *b = proyected.at(face.at(2) - 1);
			SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
			SDL_RenderDrawLine(renderer, a->x, a->y, b->x, b->y);
		}

	}
}