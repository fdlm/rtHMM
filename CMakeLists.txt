CMAKE_MINIMUM_REQUIRED (VERSION 2.8.5) 
PROJECT (rtHMM)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

ENABLE_TESTING()

MESSAGE(WARNING "
     The rtHMM library makes heavy use of C++11. Compilation is verified to
     work on Linux with g++ 4.8.2, and thus should work using g++ >= 4.8.2 on
     other systems, too. There is no nice way to check if your compiler
     supports the necessary features using CMake, so proceed with care
     ")

# enable c++11 support in gcc
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")

# try to find google test...
FIND_PACKAGE(GTest QUIET)
IF (NOT GTEST_FOUND)
    # try to find it in the 3rd_party directory
    SET(GTEST_ROOT ${CMAKE_SOURCE_DIR}/3rd_party/gtest)
    FIND_PACKAGE(GTest)
ENDIF (NOT GTEST_FOUND)

FIND_PACKAGE(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/rtHMM/)
INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

ADD_SUBDIRECTORY(rtHMM)

IF (GTEST_FOUND)
    FIND_PACKAGE(Threads REQUIRED)
    ADD_SUBDIRECTORY(test)
ELSE (GTEST_FOUND)
    MESSAGE( STATUS "Google Test Framework not found - building without tests!" )
ENDIF (GTEST_FOUND)
