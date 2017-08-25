cmake_minimum_required(VERSION 2.8)
set(project_name find_edges)
project(${project_name})

file(GLOB SOURCES "./*.cpp" "./*.h")

if (MSVC)
    add_definitions(/W4 /WX)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Werror -Wall")
endif()

find_package(OpenCV REQUIRED)
add_executable(${project_name} ${SOURCES})
target_link_libraries(${project_name} ${OpenCV_LIBS})

