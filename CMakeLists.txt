# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.13)

project ("dphpc-bvh")


include_directories(${PROJECT_SOURCE_DIR}/Include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Install google test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.8.0
)

FetchContent_GetProperties(googletest)

if(NOT googletest_POPULATED)
  	FetchContent_Populate(googletest)
		set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
		set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
		# add the targets: gtest,gtest_main,gmock,gmock_main
		add_subdirectory(
			${googletest_SOURCE_DIR}
			${googletest_BINARY_DIR}
		)
endif()


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

add_executable (main "Src/Main.cpp" ${SRC})

# TESTING
enable_testing()

add_executable(unitTestTest "Test/unitTestTest.cc")
target_link_libraries( unitTestTest gtest_main )


include(GoogleTest)
gtest_discover_tests(unitTestTest)
