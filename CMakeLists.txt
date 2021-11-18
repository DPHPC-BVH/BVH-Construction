# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("dphpc-bvh")


include_directories(${PROJECT_SOURCE_DIR}/Include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 -pthread")
set(CMAKE_C_FLAGS "-O3 -pthread")

set ( SRC 
        "Src/BVH.cpp"
        "Src/BVHBuilder.cpp"
        #"Src/CudaBVHBuilder.cpp"
        "Src/Memory.cpp"
        "Src/Parallel.cpp"
        "Src/RecursiveBVHBuilder.cpp"
        "Src/Renderer.cpp"
        "Src/Scene.cpp"
        "Src/tiny_obj_loader.cc"
        "Src/Transform.cpp"
        "Src/Triangle.cpp"
		)

add_executable (main "Test/Main.cpp" ${SRC})
