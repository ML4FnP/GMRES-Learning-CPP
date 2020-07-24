cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Main)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(Main.ex OfflineGMRES.cpp)
target_link_libraries(Main.ex "${TORCH_LIBRARIES}")