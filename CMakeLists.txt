# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.13)

project ("dphpc-bvh" LANGUAGES CXX CUDA)

# Set build type to RELEASE by default:
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to RELEASE.")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif ()

option(BVH_TEST 		"BVH_TEST" 		ON)
option(BVH_BENCH       	"BVH_BENCH"     ON)

set(BVH_LIB_NAME dphpcbvh)
set(BVH_LIB_TYPE SHARED)

# Find CUDA and later include $CUDA_INCLUDE_DIRS
find_package(CUDA 11.0 REQUIRED)
find_package (Threads REQUIRED)

# Define include directories
include_directories(${PROJECT_SOURCE_DIR}/Include ${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_C_FLAGS "-O3")


add_subdirectory(Src)

if (BVH_TEST)
	enable_testing()
	add_subdirectory(Test)
endif ()

if (BVH_BENCH)
  add_subdirectory(Bench)
endif ()