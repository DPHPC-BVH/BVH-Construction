set(bvh_src
    BVH.cpp
    BVHBuilder.cpp
    CudaBVHBuilder.cpp
    CudaBVHBuilder.cu
    CubWrapper.cu
    Memory.cpp
    Parallel.cpp
    RecursiveBVHBuilder.cpp
    Renderer.cpp
    Scene.cpp
    tiny_obj_loader.cc
    Transform.cpp
    Triangle.cpp
)

cuda_add_library(${BVH_LIB_NAME} ${BVH_LIB_TYPE} ${bvh_src})

target_link_libraries(${BVH_LIB_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS ${BVH_LIB_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ../include/ DESTINATION include)

add_executable (main Main.cpp ${SRC})
target_link_libraries(main ${BVH_LIB_NAME})
