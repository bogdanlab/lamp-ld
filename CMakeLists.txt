cmake_minimum_required(VERSION 3.17)
project(LAMP_LD)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMY_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DMY_DEBUG")

set(CMAKE_CXX_STANDARD 17)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "boost not found.")
endif()

if (NOT EIGEN3_FOUND)
    message( FATAL_ERROR "Eigen3 not found.")
endif()

message( STATUS "Eigen3 DIR: " ${EIGEN3_INCLUDE_DIR})
message( STATUS "Boost Dir: " ${Boost_INCLUDE_DIR})

include_directories(${Boost_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
add_executable(LAMP_LD src/main.cpp src/WindowHMM.cpp src/WindowHMM.h src/LampLD.cpp src/LampLD.h src/FileUtils.cpp src/FileUtils.h src/Utils.cpp src/Utils.h)

target_link_libraries(LAMP_LD ${Boost_LIBRARIES})
