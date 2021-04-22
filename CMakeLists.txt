cmake_minimum_required(VERSION 3.12)
project(lampld)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)


add_subdirectory(thirdparty/pybind11)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/)

#add_executable(lampld src/main.cpp src/WindowHMM.cpp src/WindowHMM.h src/LampLD.cpp src/LampLD.h src/FileUtils.cpp src/FileUtils.h src/Utils.cpp src/Utils.h src/pywrapper.cpp)
#install(TARGETS lampld)

pybind11_add_module(lampld src/pywrapper.cpp src/WindowHMM.cpp src/WindowHMM.h src/LampLD.cpp src/LampLD.h src/FileUtils.cpp src/FileUtils.h src/Utils.cpp src/Utils.h)