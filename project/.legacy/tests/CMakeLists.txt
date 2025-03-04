cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(catch2 LANGUAGES CXX)

add_subdirectory(catch)

# file(GLOB SOURCES "./**.cpp")

# add_executable(catch2 catch_main.cpp catch_tests_fizzbuzz)
# add_executable(catch2 ${SOURCES})



# Find Python
# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Glob all the files in src/
file(GLOB SOURCES "./**.cpp")

# Create executable
add_executable(catch2 ${SOURCES})
target_link_libraries(catch2 PRIVATE catch)

# # Find NumPy (ensure NumPy is installed in the Python environment)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Include directories for NumPy and Python
include_directories(catch 
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)