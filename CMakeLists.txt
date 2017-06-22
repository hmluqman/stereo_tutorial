cmake_minimum_required(VERSION 2.8)

project(stereo_tutorial)
find_package(OpenCV REQUIRED)

###########
## Build ##
###########
if(CMAKE_VERSION VERSION_LESS "3.1")
     if(CMAKE_COMPILER_IS_GNUCXX)
       include(CheckCXXCompilerFlag)
       check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
       check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
       if(COMPILER_SUPPORTS_CXX11)
     add_definitions("-std=c++11")
       elseif(COMPILER_SUPPORTS_CXX0X)
     add_definitions("-std=c++0x")
       else()
     message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no
C++11 support.")
       endif()
     endif()
   else()
     set(CMAKE_CXX_STANDARD 11)
endif()


include_directories(
# include
  ${OpenCV_INCLUDE_DIRS}
#  ${EIGEN_INCLUDE_DIRS}
)

add_executable(stereo_main "main.cpp")
target_link_libraries(stereo_main ${OpenCV_LIBRARIES})
