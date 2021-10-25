# CMakeList.txt : CMake project for bvh, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("bvh")

# Add source to this project's executable.
add_executable (bvh "Source/BVH.cpp" "Source/BVH.h")

# TODO: Add tests and install targets if needed.
