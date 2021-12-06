add_subdirectory(googletest)

include_directories(./googletest/googletest/include)

set(test_src
  RunAll.cc
  UnitTestTest.cc
  CubWrapperTest.cc
  CudaBVHBuilderTest.cc
)

add_executable(test_dphp_bvh ${test_src})

target_link_libraries(test_dphp_bvh
  gtest
  ${BVH_LIB_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
)

add_test(
  NAME tests
  COMMAND test_dphp_bvh
)