cmake_minimum_required(VERSION 3.0.2)
set(project_name poly_match)
project(${project_name})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall")
option(WITH_PYTHON "Build with python interface" OFF)

if (WITH_PYTHON)
    find_package(Boost 1.60 REQUIRED COMPONENTS python numpy)
    find_package(PythonLibs REQUIRED)

    include_directories(PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${PYTHON_INCLUDE_PATH} ${Boost_INCLUDE_DIRS})
endif (WITH_PYTHON)

add_library(${project_name} SHARED poly_match.cpp poly_match.hpp)
set_target_properties(${project_name} PROPERTIES PREFIX "")

file(GLOB_RECURSE UTM_SOURCE_FILES "3rd-party/utm/*")
add_library(utm SHARED ${UTM_SOURCE_FILES})

add_executable(ymaprica metric.cpp)
target_link_libraries(ymaprica ${project_name})
target_link_libraries(ymaprica utm)

if (WITH_PYTHON)
    target_link_libraries(${project_name} ${Boost_LIBRARIES})
    target_link_libraries(ymaprica ${PYTHON_LIBRARIES})
endif (WITH_PYTHON)

