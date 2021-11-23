set(test_src
  runAll.cc
  unitTestTest.cc
  cubWrapperTest.cc
)

add_executable(test_dphp_bvh ${test_src})

target_link_libraries(test_dphp_bvh
  gtest_main
  ${BVH_LIB_NAME}
)

add_test(
  NAME tests
  COMMAND test_dphp_bvh
)