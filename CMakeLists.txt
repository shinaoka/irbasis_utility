cmake_minimum_required (VERSION 2.8.12)

# use only Python
project (irbasis_utility NONE)

# Print build type
if(NOT CMAKE_BUILD_TYPE)
    message("Using default build type: Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

enable_testing(test)
add_subdirectory(test)

add_subdirectory(python)
