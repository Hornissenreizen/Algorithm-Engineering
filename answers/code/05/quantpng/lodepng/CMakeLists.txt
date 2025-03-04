cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(lodepng LANGUAGES CXX)

file(GLOB SOURCES "./*.cpp" "./*.h")

add_library(lodepng STATIC ${SOURCES})

target_include_directories(lodepng PUBLIC ./)