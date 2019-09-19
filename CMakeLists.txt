cmake_minimum_required(VERSION 2.6)
set(META_PROJECT_NAME        "CUDAStudy")  #对应文件AliceOperationAgent-config.cmake

project(${META_PROJECT_NAME})

# Include cmake modules

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(source)