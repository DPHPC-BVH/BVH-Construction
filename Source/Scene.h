#pragma once

#include "DPHPC.h"
#include "Geometry.h"
#include "Triangle.h"
#include "BVH.h"


NAMESPACE_DPHPC_BEGIN
class Scene {
public:
	void LoadMesh(std::string path);
	Scene();

private:
	int numTriangles;
	int numVertices;

	std::vector<Point3f> vertexBuffer;
	std::vector<int> indexBuffer;
	std::vector<Triangle> triangles;

	BVH bvh;

};


NAMESPACE_DPHPC_END


