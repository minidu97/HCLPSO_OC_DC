cmake_minimum_required(VERSION 3.16)
project(HCLPSO_OC_DC LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -O2")

add_executable(hclpso
    main.cpp
    cec2017/cec17.c
    cec2017/cec17_test_func.c
)

target_include_directories(hclpso PRIVATE
    ${CMAKE_SOURCE_DIR}/cec2017
)
