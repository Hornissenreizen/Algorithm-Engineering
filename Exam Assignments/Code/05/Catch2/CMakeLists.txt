cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(catch2 LANGUAGES CXX)

add_subdirectory(catch)
add_subdirectory(fizzbuzz)

add_executable(catch2 catch_main.cpp catch_tests_fizzbuzz)

target_link_libraries(catch2 PRIVATE catch fizzbuzz)