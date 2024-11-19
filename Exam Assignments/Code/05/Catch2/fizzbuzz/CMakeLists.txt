cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(fizzbuzz LANGUAGES CXX)

file(GLOB SOURCES "./**/*.cpp" "./**/*.h")

add_library(fizzbuzz STATIC ${SOURCES})

target_include_directories(fizzbuzz PUBLIC include)