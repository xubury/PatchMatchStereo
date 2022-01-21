cmake_minimum_required(VERSION 3.1)
project(PatchMatch)


find_package(glm REQUIRED)
find_package(OpenCV REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/release)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/release)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/release)
endif()

add_executable(PatchMatch 
    src/main.cpp 
    src/PatchMatchStereo.cpp
    src/PMSPropagation.cpp)

target_compile_options(PatchMatch PUBLIC -Wall -Wextra -Wundef -pedantic -std=c++17)

target_link_libraries(PatchMatch ${OpenCV_LIBS} glm)
target_include_directories(PatchMatch PUBLIC ${OpenCV_INCLUDE_DIRS} include)
