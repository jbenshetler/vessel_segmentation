cmake_minimum_required(VERSION 3.20.3)
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD) 
endif()

set(CMAKE_CXX_STANDARD 20)

project(vessel_segmentation)


find_package(OpenCV 4.8 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable ( vessel_segmentation vessel_segmentation.cpp )
target_link_libraries ( vessel_segmentation ${SimpleITK_LIBRARIES} )
target_link_libraries( vessel_segmentation ${OpenCV_LIBS} )