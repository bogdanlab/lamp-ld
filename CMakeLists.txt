cmake_minimum_required(VERSION 3.4)
project(LAMP_LD)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)

add_executable(lampld src/main.cpp src/WindowHMM.cpp src/WindowHMM.h src/LampLD.cpp src/LampLD.h src/FileUtils.cpp src/FileUtils.h src/Utils.cpp src/Utils.h)

install(TARGETS lampld)

