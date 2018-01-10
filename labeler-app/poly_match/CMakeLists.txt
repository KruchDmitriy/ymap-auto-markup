cmake_minimum_required(VERSION 3.0.2)
set(proect_name poly_match)
project(${proect_name})

set(CMAKE_CXX_FLAGS "-Wall")

find_package(Boost 1.60 REQUIRED COMPONENTS python numpy)
find_package(PythonLibs REQUIRED)

include_directories(PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PYTHON_INCLUDE_PATH} ${Boost_INCLUDE_DIRS})

add_library(${proect_name} SHARED poly_match.cpp poly_match.hpp)
set_target_properties(${proect_name} PROPERTIES PREFIX "")
target_link_libraries(${proect_name} ${Boost_LIBRARIES})
