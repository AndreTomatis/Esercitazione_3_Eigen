cmake_minimum_required(VERSION 3.9)

project(SingularMatrix LANGUAGES CXX C VERSION 1.0.0)

# Set C++ Standard
################################################################################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# IMPOSE WARNINGS ON DEBUG
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -pedantic-errors")

# IMPOSE CXX FLAGS FOR WINDOWS
if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
endif (WIN32)

# Add dependencies
################################################################################

## Eigen3
find_package(Eigen3 CONFIG REQUIRED)
list(APPEND SingularMatrix_LINKED_LIBRARIES PUBLIC Eigen3::Eigen)

## Threads
find_package(Threads REQUIRED)
list(APPEND SingularMatrix_LINKED_LIBRARIES PRIVATE Threads::Threads)

## GTest
find_package(GTest REQUIRED)

list(APPEND SingularMatrix_LINKED_LIBRARIES PRIVATE GTest::gtest)
list(APPEND SingularMatrix_LINKED_LIBRARIES PRIVATE GTest::gtest_main)
list(APPEND SingularMatrix_LINKED_LIBRARIES PRIVATE GTest::gmock)
list(APPEND SingularMatrix_LINKED_LIBRARIES PRIVATE GTest::gmock_main)

# Insert Sources
################################################################################

list(APPEND SingularMatrix_SOURCES ${SingularMatrix_sources})
list(APPEND SingularMatrix_HEADERS ${SingularMatrix_headers})
list(APPEND SingularMatrix_INCLUDE ${SingularMatrix_includes})

# Create executable
################################################################################
add_executable(${PROJECT_NAME}
    main.cpp
    ${SingularMatrix_SOURCES}
    ${SingularMatrix_HEADERS})

target_link_libraries(${PROJECT_NAME} ${SingularMatrix_LINKED_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${SingularMatrix_INCLUDE})
target_compile_options(${PROJECT_NAME} PUBLIC -fPIC)
