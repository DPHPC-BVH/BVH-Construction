#pragma once

#include "DPHPC.h"
#include "Geometry.h"
#include "Triangle.h"
#include "BVH.h"
#include "BVHBuilder.h"


NAMESPACE_DPHPC_BEGIN
class Scene {
	friend class Renderer;
public:
	void LoadMesh(std::string path, BVHBuilderType type);
	Scene();

private:
	void LoadMeshFromFile(std::string path);
	void BuildBVH(BVHBuilderType type);
	
	int numTriangles;
	int numVertices;

	std::vector<Point3f> vertexBuffer;
	std::vector<int> indexBuffer;
	std::vector<Triangle> triangles;

	BVH bvh;

};


NAMESPACE_DPHPC_END


