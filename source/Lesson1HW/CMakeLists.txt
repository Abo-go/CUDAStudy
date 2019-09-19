cmake_minimum_required(VERSION 2.6)

set(target        "Lesson1HW")  #对应文件AliceOperationAgent-config.cmake
#// cmake寻找cuda，并且CUDA_ADD_EXECUTABLE等cuda前缀的命令是由cuda提供的，cmake本身不支持
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpencvWorld REQUIRED)

if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

enable_language(CUDA) 

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/source")
    
file(GLOB headers
    ${include_path}/*.h
    ${include_path}/*.cuh
    )
    
file(GLOB sources
    ${source_path}/*.cpp
    ${source_path}/*.cu
    )
    
#add_executable(${target} ${sources} ${headers})
#// 注意cmake调用nvcc进行编译
CUDA_ADD_EXECUTABLE( ${target} ${sources} ${headers})

target_include_directories(${target}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OPENCVWORLD_INCLUDE_DIR}
        )
        
target_link_libraries(${target}
    PRIVATE
        ${OPENCVWORLD_LIBRARY_debug}
        )