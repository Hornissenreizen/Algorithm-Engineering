cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(catch2 LANGUAGES CXX)

# Add the catch subdirectory
add_subdirectory(catch)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Add the executable
add_executable(catch2 catch_main.cpp catch_tests_fizzbuzz.cpp)

# Link libraries
target_link_libraries(catch2 
    PRIVATE catch       # Link Catch2
    PRIVATE einspeed    # Link your einspeed library
    PRIVATE Python3::Python # Link Python library
)
