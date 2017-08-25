cmake_minimum_required(VERSION 2.8)
set(project_name find_edges)
project(${project_name})

file(GLOB SOURCES "./*.cpp" "./*.h")

if (MSVC)
    add_definitions(/W4 /WX)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Werror -Wall")
endif()



add_executable(${project_name} ${SOURCES})

find_package(OpenCV REQUIRED)
target_link_libraries(${project_name} ${OpenCV_LIBS})

set(Boost_USE_STATIC_LIBS        ON) # only find static libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

find_package(Boost REQUIRED)

target_include_directories(${project_name} PRIVATE ${Boost_INCLUDEDIR})
#target_link_libraries(${project_name} ${Boost_UBLAS_LIBRARY_RELEASE})
