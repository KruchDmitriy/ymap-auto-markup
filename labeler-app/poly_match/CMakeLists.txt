cmake_minimum_required(VERSION 3.0.2)
set(project_name poly_match)
project(${project_name})

set(CMAKE_CXX_FLAGS "-Wall")

if (WITH_PYTHON)
    find_package(Boost 1.60 REQUIRED COMPONENTS python numpy)
    find_package(PythonLibs REQUIRED)

    include_directories(PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${PYTHON_INCLUDE_PATH} ${Boost_INCLUDE_DIRS})
endif (WITH_PYTHON)

add_library(${project_name} SHARED poly_match.cpp poly_match.hpp)
set_target_properties(${project_name} PROPERTIES PREFIX "")

add_executable(ymaprica metric.cpp)
target_link_libraries(ymaprica poly_match)

if (WITH_PYTHON)
    target_link_libraries(${project_name} ${Boost_LIBRARIES})
    target_link_libraries(ymaprica ${Boost_LIBRARIES})
endif (WITH_PYTHON)

