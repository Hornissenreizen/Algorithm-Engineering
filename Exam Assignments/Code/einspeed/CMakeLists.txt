cmake_minimum_required(VERSION 3.10)

project(main VERSION 1.0 LANGUAGES CXX)

include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/external/hptt/include)
add_subdirectory(src)
add_subdirectory(tests)