cmake_minimum_required(VERSION 3.17)
project(LAMP_LD)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DMY_DEBUG")

set(CMAKE_CXX_STANDARD 17)

add_executable(LAMP_LD src/main.cpp src/WindowHMM.cpp src/WindowHMM.h src/LampLD.cpp src/LampLD.h src/FileUtils.cpp src/FileUtils.h src/Utils.cpp src/Utils.h)

target_link_libraries(LAMP_LD)
