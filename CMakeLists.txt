cmake_minimum_required(VERSION 2.8.7)
project(isee_mscan_reid)

option(test OFF)
if (test)
    enable_testing()
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set (JAVA_HOME $ENV{JAVA_HOME} CACHE PATH "Home directory of Java Divelooment Toolkit (JDK).")

link_directories(${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(jni lib/jni)

set(TEST_VIDEO_PATH CACHE FILEPATH "File path of video for test.")
#add_subdirectory(test bin)

install (FILES
        ${PROJECT_SOURCE_DIR}/lib/libisee_mscan_reid.so
        DESTINATION
        lib)
install (FILES
        ${PROJECT_SOURCE_DIR}/include/isee_mscan_reid.hpp
        DESTINATION
        include/isee_mscan_reid)
install (FILES
        ${PROJECT_SOURCE_DIR}/include/attributes.h
        DESTINATION
        include/isee_mscan_reid)
