#include "Scene.h"
#include "tiny_obj_loader.h"
#include <vector>
#include <string>
#include <iostream>
#include "RecursiveBVHBuilder.h"
#include "CudaBVHBuilder.h"
#include "BVHBuilder.h"

NAMESPACE_DPHPC_BEGIN
using namespace tinyobj;


Scene::Scene() :
	numTriangles(0),
	numVertices(0)

{

}




void Scene::LoadMesh(std::string path, BVHBuilderType type) {
	Scene::LoadMeshFromFile(path);
	Scene::BuildBVH(type);
}

void Scene::LoadMeshFromFile(std::string path) {
	attrib_t attribute;
	std::vector<shape_t> shapes;
	std::vector<material_t> materials;
	std::string warning;
	std::string error;
	tinyobj::LoadObj(&attribute, &shapes, &materials, &warning, &error, path.c_str(), nullptr, true/*with triangulation*/);

	int numIndices = 0;
	numVertices = attribute.vertices.size() / 3;
	vertexBuffer.reserve(numVertices);
	for (int i = 0; i < numVertices; ++i) {
		vertexBuffer.push_back(Point3f(attribute.vertices[3*i], attribute.vertices[3*i+1], attribute.vertices[3*i+2]));
	}
	for (auto& shape : shapes) {
		numIndices += shape.mesh.indices.size();
		numTriangles += shape.mesh.num_face_vertices.size();
		for (int i = 0; i < shape.mesh.indices.size(); ++i) {
			indexBuffer.push_back(shape.mesh.indices[i].vertex_index);
		}
	}

	CHECK(numTriangles * 3 == numIndices);  // Guarantee all faces are triangles

	triangles.reserve(numTriangles);
	for (int i = 0; i < numTriangles; ++i) {
		triangles.push_back(Triangle(i, vertexBuffer.data(), indexBuffer.data()));
	}
}

void Scene::BuildBVH(BVHBuilderType type) {

	std::vector<std::shared_ptr<Primitive>> pTriangles;
	pTriangles.reserve(numTriangles);
	for (int i = 0; i < numTriangles; ++i) {
		pTriangles.push_back(std::make_shared<Triangle>(triangles[i]));
	}

	bvh = BVH(pTriangles);
	BVHBuilder* builder = BVHBuilder::MakeBVHBuilder(type, &bvh);
	builder->BuildBVH();
}



NAMESPACE_DPHPC_END

