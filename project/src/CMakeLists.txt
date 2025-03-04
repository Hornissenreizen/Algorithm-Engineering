cmake_minimum_required(VERSION 3.10)

# Project name and version
project(einspeed VERSION 1.0 LANGUAGES CXX)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -ffast-math")

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Glob all the files in src/
file(GLOB SOURCES "./**.cpp")

# Create library
# add_library(einspeed MODULE ${SOURCES})
add_library(einspeed SHARED ${SOURCES})

# Add the external library hptt
add_subdirectory(${CMAKE_SOURCE_DIR}/external/hptt ${CMAKE_BINARY_DIR}/external/hptt)
target_link_libraries(einspeed PUBLIC hptt)



# Find NumPy (ensure NumPy is installed in the Python environment)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the library output name
set_target_properties(einspeed PROPERTIES
    PREFIX ""  # Required for Python extension modules
    SUFFIX ".so"
)

# Include directories for NumPy and Python
target_include_directories(einspeed PUBLIC
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)