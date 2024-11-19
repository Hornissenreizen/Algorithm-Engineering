cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(quantize LANGUAGES C)

file(GLOB SOURCES "./*.c" "./*.h")

add_library(quantize STATIC ${SOURCES})

target_include_directories(quantize PUBLIC ./)