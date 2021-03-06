#include <gtest/gtest.h>
#include "CudaBVHBuilder.cuh"
#include "CudaBVHBuilder.h"

NAMESPACE_DPHPC_BEGIN

TEST(CudaBVHBuilderTest, LeftShiftAndExpand32) {

  EXPECT_EQ(LeftShiftAndExpand32(0b00000000000000000000000101010011), 0b00000001000001000001000000001001);
  EXPECT_EQ(LeftShiftAndExpand32(0b00000000000000000000001111111111), 0b00001001001001001001001001001001);
  EXPECT_EQ(LeftShiftAndExpand32(0b00000000000000000000000111001011), 0b00000001001001000000001000001001);

}

TEST(CudaBVHBuilderTest, getMortonCode32) {

  float x = 0.3310546875f; // 0b00000000000000000000000101010011 (339)  / 1024 = 0.3310546875
  float y = 0.9990234375f; // 0b00000000000000000000001111111111 (1023) / 1024 = 0.9990234375
  float z = 0.4482421875f; // 0b00000000000000000000000111001011 (459)  / 1024 = 0.4482421875

  EXPECT_EQ(GetMortonCode32(x, y, z), 0b00010111110111010011110010111111);

  x = 0.3310546875398475f; // 0b00000000000000000000000101010011 (339)  / 1024 = 0.3310546875
  y = 0.9990234375893247f; // 0b00000000000000000000001111111111 (1023) / 1024 = 0.9990234375
  z = 0.4482421875983744f; // 0b00000000000000000000000111001011 (459)  / 1024 = 0.4482421875

  EXPECT_EQ(GetMortonCode32(x, y, z), 0b00010111110111010011110010111111);

}

TEST(CudaBVHBuilderTest, BuildTreeHierarchy) {

  constexpr unsigned int nPrimitives = 8;
  unsigned int mortonCodeSorted[nPrimitives] = {
      0b00000000000000000000000000000001,
      0b00000000000000000000000000000010,
      0b00000000000000000000000000000100,
      0b00000000000000000000000000000101,
      0b00000000000000000000000000010011,
      0b00000000000000000000000000011000,
      0b00000000000000000000000000011001,
      0b00000000000000000000000000011110,
  };

  unsigned int indicesSorted[nPrimitives] = {6, 3, 7, 1, 0, 5, 2, 4};

  CudaBVHBuildNode treeExpected[2*nPrimitives - 1] = {
    // Interior nodes
    CudaBVHBuildNode(3, 4, -1),
    CudaBVHBuildNode(7, 8, 3),
    CudaBVHBuildNode(9, 10, 3),
    CudaBVHBuildNode(1, 2, 0),
    CudaBVHBuildNode(11, 5, 0),
    CudaBVHBuildNode(6, 14, 4),
    CudaBVHBuildNode(12, 13, 5),

    // Leafs
    CudaBVHBuildNode(-1, -1, 1, indicesSorted[0]),
    CudaBVHBuildNode(-1, -1, 1, indicesSorted[1]),
    CudaBVHBuildNode(-1, -1, 2, indicesSorted[2]),
    CudaBVHBuildNode(-1, -1, 2, indicesSorted[3]),
    CudaBVHBuildNode(-1, -1, 4, indicesSorted[4]),
    CudaBVHBuildNode(-1, -1, 6, indicesSorted[5]),
    CudaBVHBuildNode(-1, -1, 6, indicesSorted[6]),
    CudaBVHBuildNode(-1, -1, 5, indicesSorted[7])

  };

  unsigned int* dMortonCodeSorted;
  cudaMalloc(&dMortonCodeSorted, nPrimitives * sizeof(unsigned int));
  cudaMemcpy(dMortonCodeSorted, mortonCodeSorted, sizeof(unsigned int) * nPrimitives, cudaMemcpyHostToDevice);

  unsigned int* dIndicesSorted;
  cudaMalloc(&dIndicesSorted, nPrimitives * sizeof(unsigned int));
  cudaMemcpy(dIndicesSorted, indicesSorted, sizeof(unsigned int) * nPrimitives, cudaMemcpyHostToDevice);

  CudaBVHBuildNode* dTree;
  cudaMalloc(&dTree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode));

  BuildTreeHierarchy(nPrimitives, dMortonCodeSorted, dIndicesSorted, dTree);

  CudaBVHBuildNode tree[2*nPrimitives - 1];
  cudaMemcpy(tree, dTree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode), cudaMemcpyDeviceToHost);

  cudaFree(dMortonCodeSorted);
  cudaFree(dIndicesSorted);
  cudaFree(dTree);

  for (size_t i = 0; i < 2 * nPrimitives - 1; i++) {
    EXPECT_EQ(tree[i].children[0], treeExpected[i].children[0]);
    EXPECT_EQ(tree[i].children[1], treeExpected[i].children[1]);
    EXPECT_EQ(tree[i].parent, treeExpected[i].parent);

    if(i < nPrimitives - 1) {
      // Interior node so dataIdx is -1
      EXPECT_EQ(tree[i].dataIdx, -1);
    }
    if(i >= nPrimitives - 1) {
      // Leaf node is dataIdx should be in [0, nPrimitives)
      EXPECT_EQ(tree[i].dataIdx, treeExpected[i].dataIdx);
    }
  }
}

TEST(CudaBVHBuilderTest, ComputeBoundingBoxes) {

  constexpr unsigned int nPrimitives = 8;
  unsigned int mortonCodeSorted[nPrimitives] = {
      0b00000000000000000000000000000001,
      0b00000000000000000000000000000010,
      0b00000000000000000000000000000100,
      0b00000000000000000000000000000101,
      0b00000000000000000000000000010011,
      0b00000000000000000000000000011000,
      0b00000000000000000000000000011001,
      0b00000000000000000000000000011110,
  };

  unsigned int indicesSorted[nPrimitives] = {6, 3, 7, 1, 0, 5, 2, 4};

  CudaBVHBuildNode tree[2*nPrimitives - 1] = {
    // Interior nodes
    CudaBVHBuildNode(3, 4, -1),
    CudaBVHBuildNode(7, 8, 3),
    CudaBVHBuildNode(9, 10, 3),
    CudaBVHBuildNode(1, 2, 0),
    CudaBVHBuildNode(11, 5, 0),
    CudaBVHBuildNode(6, 14, 4),
    CudaBVHBuildNode(12, 13, 5),

    // Leafs
    CudaBVHBuildNode(-1, -1, 1, indicesSorted[0]),
    CudaBVHBuildNode(-1, -1, 1, indicesSorted[1]),
    CudaBVHBuildNode(-1, -1, 2, indicesSorted[2]),
    CudaBVHBuildNode(-1, -1, 2, indicesSorted[3]),
    CudaBVHBuildNode(-1, -1, 4, indicesSorted[4]),
    CudaBVHBuildNode(-1, -1, 6, indicesSorted[5]),
    CudaBVHBuildNode(-1, -1, 6, indicesSorted[6]),
    CudaBVHBuildNode(-1, -1, 5, indicesSorted[7])

  };

  auto randomFloat = []() {return static_cast <Float> (rand()) / static_cast <Float> (RAND_MAX); };

  BVHPrimitiveInfoWithIndex primitiveInfo[nPrimitives] = {
    BVHPrimitiveInfoWithIndex(indicesSorted[0], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[1], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[2], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[3], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[4], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[5], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[6], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[7], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat())))
  };

  
  CudaBVHBuildNode* dTree;
  cudaMalloc(&dTree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode));
  cudaMemcpy(dTree, tree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode), cudaMemcpyHostToDevice);

  BVHPrimitiveInfoWithIndex* dPrimitiveInfo;
  cudaMalloc(&dPrimitiveInfo, nPrimitives * sizeof(BVHPrimitiveInfoWithIndex));
  cudaMemcpy(dPrimitiveInfo, primitiveInfo, nPrimitives * sizeof(BVHPrimitiveInfoWithIndex), cudaMemcpyHostToDevice);

  ComputeBoundingBoxes(nPrimitives, dTree, dPrimitiveInfo);

  CudaBVHBuildNode treeWithBoundingBoxes[2*nPrimitives - 1];
  cudaMemcpy(treeWithBoundingBoxes, dTree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode), cudaMemcpyDeviceToHost);

  cudaFree(dTree);
  cudaFree(dPrimitiveInfo);
  

  for (size_t i = 0; i < 2 * nPrimitives - 1; i++) {

    if(i < nPrimitives - 1) {
      // Interior node so dataIdx is -1
      CudaBVHBuildNode leftChild = treeWithBoundingBoxes[treeWithBoundingBoxes[i].children[0]];
      CudaBVHBuildNode rightChild = treeWithBoundingBoxes[treeWithBoundingBoxes[i].children[1]];

      EXPECT_EQ(treeWithBoundingBoxes[i].bounds, Union(leftChild.bounds, rightChild.bounds));
    }
    if(i >= nPrimitives - 1) {
      // Leaf node is dataIdx should be in [0, nPrimitives)
      EXPECT_EQ(treeWithBoundingBoxes[i].bounds, primitiveInfo[treeWithBoundingBoxes[i].dataIdx].bounds);
    }
  }

}

TEST(CudaBVHBuilderTest, ComputeBoundingBoxesWithSharedMemory) {

  constexpr unsigned int nPrimitives = 8;
  unsigned int mortonCodeSorted[nPrimitives] = {
      0b00000000000000000000000000000001,
      0b00000000000000000000000000000010,
      0b00000000000000000000000000000100,
      0b00000000000000000000000000000101,
      0b00000000000000000000000000010011,
      0b00000000000000000000000000011000,
      0b00000000000000000000000000011001,
      0b00000000000000000000000000011110,
  };

  unsigned int indicesSorted[nPrimitives] = {6, 3, 7, 1, 0, 5, 2, 4};

  CudaBVHBuildNode tree[2*nPrimitives - 1] = {
    // Interior nodes
    CudaBVHBuildNode(3, 4, -1),
    CudaBVHBuildNode(7, 8, 3),
    CudaBVHBuildNode(9, 10, 3),
    CudaBVHBuildNode(1, 2, 0),
    CudaBVHBuildNode(11, 5, 0),
    CudaBVHBuildNode(6, 14, 4),
    CudaBVHBuildNode(12, 13, 5),

    // Leafs
    CudaBVHBuildNode(-1, -1, 1, indicesSorted[0]),
    CudaBVHBuildNode(-1, -1, 1, indicesSorted[1]),
    CudaBVHBuildNode(-1, -1, 2, indicesSorted[2]),
    CudaBVHBuildNode(-1, -1, 2, indicesSorted[3]),
    CudaBVHBuildNode(-1, -1, 4, indicesSorted[4]),
    CudaBVHBuildNode(-1, -1, 6, indicesSorted[5]),
    CudaBVHBuildNode(-1, -1, 6, indicesSorted[6]),
    CudaBVHBuildNode(-1, -1, 5, indicesSorted[7])

  };

  auto randomFloat = []() {return static_cast <Float> (rand()) / static_cast <Float> (RAND_MAX); };

  BVHPrimitiveInfoWithIndex primitiveInfo[nPrimitives] = {
    BVHPrimitiveInfoWithIndex(indicesSorted[0], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[1], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[2], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[3], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[4], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[5], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[6], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat()))),
    BVHPrimitiveInfoWithIndex(indicesSorted[7], Bounds3f(Point3f(randomFloat(), randomFloat(), randomFloat()), Point3f(randomFloat(), randomFloat(), randomFloat())))
  };

  
  CudaBVHBuildNode* dTree;
  cudaMalloc(&dTree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode));
  cudaMemcpy(dTree, tree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode), cudaMemcpyHostToDevice);

  BVHPrimitiveInfoWithIndex* dPrimitiveInfo;
  cudaMalloc(&dPrimitiveInfo, nPrimitives * sizeof(BVHPrimitiveInfoWithIndex));
  cudaMemcpy(dPrimitiveInfo, primitiveInfo, nPrimitives * sizeof(BVHPrimitiveInfoWithIndex), cudaMemcpyHostToDevice);

  ComputeBoundingBoxesWithSharedMemory<2>(nPrimitives, dTree, dPrimitiveInfo);

  CudaBVHBuildNode treeWithBoundingBoxes[2*nPrimitives - 1];
  cudaMemcpy(treeWithBoundingBoxes, dTree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode), cudaMemcpyDeviceToHost);

  cudaFree(dTree);
  cudaFree(dPrimitiveInfo);
  

  for (size_t i = 0; i < 2 * nPrimitives - 1; i++) {

    if(i < nPrimitives - 1) {
      // Interior node so dataIdx is -1
      CudaBVHBuildNode leftChild = treeWithBoundingBoxes[treeWithBoundingBoxes[i].children[0]];
      CudaBVHBuildNode rightChild = treeWithBoundingBoxes[treeWithBoundingBoxes[i].children[1]];

      EXPECT_EQ(treeWithBoundingBoxes[i].bounds, Union(leftChild.bounds, rightChild.bounds));
    }
    if(i >= nPrimitives - 1) {
      // Leaf node is dataIdx should be in [0, nPrimitives)
      EXPECT_EQ(treeWithBoundingBoxes[i].bounds, primitiveInfo[treeWithBoundingBoxes[i].dataIdx].bounds);
    }
  }

}
NAMESPACE_DPHPC_END