cmake_minimum_required(VERSION 3.12)
project(lampld)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/thirdparty/)
set(SOURCE_FILES src/WindowHMM.cpp src/WindowHMM.h src/LampLD.cpp src/LampLD.h src/FileUtils.cpp src/FileUtils.h src/Utils.cpp src/Utils.h)


if(NOT BUILD_EXE)
    add_subdirectory(thirdparty/pybind11)
    pybind11_add_module(pylampld src/pywrapper.cpp ${SOURCE_FILES})
else()
    add_executable(lampld ../src/main.cpp ${SOURCE_FILES} )
    install(TARGETS lampld)
endif()