#include <gtest/gtest.h>
#include <vector>

#include "CubWrapper.cuh"


NAMESPACE_DPHPC_BEGIN

TEST(CubWrapperTest, DeviceSort) {
  
  const int numItems = 7;
  unsigned int keysIn[7] = {8, 6, 7, 5, 3, 0, 9};
  unsigned int valuesIn[7] = {0, 1, 2, 3, 4, 5, 6};
  unsigned int keysOut[7];
  unsigned int valuesOut[7];

  DeviceSort(numItems, keysIn, keysOut, valuesIn, valuesOut);
  
  unsigned int keysOutExpected[7] = {0, 3, 5, 6, 7, 8, 9};
  for (size_t i = 0; i < numItems; i++)
    EXPECT_EQ(keysOut[i], keysOutExpected[i]);
  
  unsigned int valuesOutExpected[7] = {5, 4, 3, 1, 2, 0, 6};
  for (size_t i = 0; i < numItems; i++)
    EXPECT_EQ(valuesOut[i], valuesOutExpected[i]);
  
}


NAMESPACE_DPHPC_END