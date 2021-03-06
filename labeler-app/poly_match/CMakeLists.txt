cmake_minimum_required(VERSION 3.0.2)
set(project_name poly_match)
project(${project_name})

set(CMAKE_CXX_STANDARD 11)
option(WITH_PYTHON "Build with python interface" OFF)

if (WITH_PYTHON)
    add_definitions(-DWITH_PYTHON)

    find_package(Boost 1.60 REQUIRED COMPONENTS python numpy)
    find_package(PythonLibs REQUIRED)

    include_directories(PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(PUBLIC ${PYTHON_INCLUDE_PATH} ${Boost_INCLUDE_DIRS})
endif (WITH_PYTHON)

add_library(${project_name} SHARED poly_match.cpp poly_match.hpp)
set_target_properties(${project_name} PROPERTIES PREFIX "")

file(GLOB_RECURSE UTM_SOURCE_FILES "3rd-party/utm/*")
add_library(utm SHARED ${UTM_SOURCE_FILES})

include_directories("3rd-party/LightGBM/include")
add_subdirectory("${PROJECT_SOURCE_DIR}/3rd-party/LightGBM" EXCLUDE_FROM_ALL)

add_executable(ymaprica metric.cpp)
target_link_libraries(ymaprica ${project_name})
target_link_libraries(ymaprica utm)
#target_link_libraries(ymaprica _lightgbm)

add_executable(metric_test test.cpp 3rd-party/catch.h)
target_link_libraries(metric_test ${project_name})
target_link_libraries(metric_test utm)

if (WITH_PYTHON)
    target_link_libraries(${project_name} ${Boost_LIBRARIES})
    target_link_libraries(ymaprica ${PYTHON_LIBRARIES})
    target_link_libraries(metric_test ${PYTHON_LIBRARIES})
endif (WITH_PYTHON)

