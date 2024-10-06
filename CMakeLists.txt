cmake_minimum_required(VERSION 3.10)

# Project name and specify C++ standard
project(KalmanFilterExample CXX)

# Set C++ standard to C++11 or higher
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen library (assumes it is installed on the system)
find_package(Eigen3 3.3 REQUIRED)

# Source file
set(SOURCE_FILES kalman_filter.cxx) # Replace with the actual file name

# Add executable target
add_executable(kalman_filter.EXE ${SOURCE_FILES})

# Link Eigen library
target_link_libraries(kalman_filter.EXE Eigen3::Eigen)
