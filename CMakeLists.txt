cmake_minimum_required(VERSION 3.25)
project(game_engine VERSION 0.1.0 LANGUAGES CXX)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(BUILD_DIR ${ROOT_DIR}/build/debug)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(BUILD_DIR ${ROOT_DIR}/build/release)
endif()

# glfw
set(GLFW_SRC_DIR ${ROOT_DIR}/deps/glfw)
set(GLFW_BUILD_DIR ${BUILD_DIR}/glfw)
add_subdirectory(${GLFW_SRC_DIR} ${GLFW_BUILD_DIR})
