# This file is in the 'lib' directory
# Similar setup for the 'lib' directory, if needed

cmake_minimum_required(VERSION 3.10)

# Project name and version
project(lib VERSION 1.0 LANGUAGES CXX)

file(GLOB SOURCES "./**.cpp")
add_library(lib STATIC ${SOURCES})