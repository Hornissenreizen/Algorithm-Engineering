cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(quantpng LANGUAGES CXX)

add_subdirectory(lodepng)
add_subdirectory(quantize)

add_executable(quantpng main.cpp)

target_link_libraries(quantpng PUBLIC lodepng quantize)