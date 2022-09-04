#include "Model.h"

#include <iostream>
#include <fstream>
#include <string>

#include <boost/algorithm/string.hpp>

/*
Vector location;
Euler rotation;

vector<Vector> vertices;
vector<int> edges;
int* faces;
Vector* normals;
*/

void parse_face(Model* model, std::vector<std::string> x,
                              std::vector<std::string> y,
                              std::vector<std::string> z )
{

    if (x.size() >= 1 && y.size() >= 1 && z.size() >= 1) {

        int vx = atoi(x.at(0).c_str());
        int vy = atoi(y.at(0).c_str());
        int vz = atoi(z.at(0).c_str());

        std::vector<int> vertex{ vx, vy, vz };
        model->faces_vertex->push_back(vertex);

        if (x.size() == 3 && y.size() == 3 && z.size() == 3) {

            int nx = atoi(x.at(2).c_str());
            int ny = atoi(y.at(2).c_str());
            int nz = atoi(z.at(2).c_str());

            std::vector<int> normal{ nx, ny, nz };
            model->faces_normal->push_back(normal);

            if (x.at(1).empty() || y.at(1).empty() || z.at(1).empty()) { // v//n

            }
            else { // v/t/n

            }
        }

        if (x.size() == 2 && y.size() == 2 && z.size() == 2) { // v/t

        }
    }

}

void parse(std::string line, Model* model) {
    std::vector<std::string> expr;
    boost::split(expr, line, boost::is_any_of(" "));
    
    // Vértices
    if (expr.at(0) == "v") {
        float x = atof(expr.at(1).c_str());
        float y = atof(expr.at(2).c_str());
        float z = atof(expr.at(3).c_str());

        Vector* vec = new Vector(x, y, z);
        model->vertices->push_back(vec);
    }

    // Faces
    if (expr.at(0) == "f") {
        std::string A = expr.at(1);
        std::string B = expr.at(2);
        std::string C = expr.at(3);

        std::vector<std::string> x;
        boost::split(x, A, boost::is_any_of("/"));
        
        std::vector<std::string> y;
        boost::split(y, B, boost::is_any_of("/"));

        std::vector<std::string> z;
        boost::split(z, C, boost::is_any_of("/"));
        
        parse_face(model, x, y, z);
    }

    // Normal
    if (expr.at(0) == "vn") {
        float x = atof(expr.at(1).c_str());
        float y = atof(expr.at(2).c_str());
        float z = atof(expr.at(3).c_str());

        Vector* vec = new Vector(x, y, z);
        model->normals->push_back(vec);
    }
}

Model::Model(std::string obj)
{
	location = Vector(25,25,0);
	rotation = Euler(0,0,0);
    scale = Vector(1, 1, 1);

    this->normals = new std::vector<Vector*>();
    this->vertices = new std::vector<Vector*>();
    this->faces_vertex = new std::vector<std::vector<int>>();
    this->faces_normal = new std::vector<std::vector<int>>();

    std::string line;
    std::ifstream myfile(obj);
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {

            if (line.find("#") == 0) continue;

            parse(line, this);
        }
        myfile.close();
    }

    else std::cout << "Unable to open file";


}
