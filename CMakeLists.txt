# CMake Project file for GPU Exponential Integrator Project
# Rafael Marinheiro (rf356@cs.cornell.edu)
# Copyright @ September 2014
project(gpu-exponential-integrator)

cmake_minimum_required(VERSION 2.6)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# General Configuration: find all the required libraries.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# compile option
option(USE_DEBUG "Turn on the debug mode" OFF)
option(USE_64BIT_ARCH "Compile the 64bit executable" ON)
option(USE_CUDA "Use CUDA" ON)
option(BUILD_LIB "Turn on to build tests" ON)
option(BUILD_TESTS "Turn on to build tests" OFF)
option(USE_MKL_BLAS             "Get BLAS/LAPACK from MKL" ON)

#===================================================================
find_package(Magma REQUIRED)





if ( USE_MKL_BLAS )
  # check for Intel MKL
  find_package(MKL REQUIRED)
  set(BLAS_LAPACK_LIBS mkl_intel_lp64 mkl_intel_thread mkl_core)
  # add_definitions(-DUSE_MKL)
  # add_definitions(-DUSE_LAPACKE)
else ()
  # Try using default libraries
  set(BLAS_LAPACK_LIBS blas lapack)
endif ()

#===================================================================
## Compiler
# set compiler flags for debug/release
if ( USE_DEBUG )
    MESSAGE(STATUS "Using debug mode")
    add_definitions(-DDEBUG)
    set(CMAKE_BUILD_TYPE Debug)
else ( USE_DEBUG )
    MESSAGE(STATUS "Using release mode")
    set(CMAKE_BUILD_TYPE Release)
endif ( USE_DEBUG )

if ( USE_64BIT_ARCH )
    add_definitions(-m64)
    set(LINK_FLAGS -m64)
endif ( USE_64BIT_ARCH )

if ( USE_DEBUG )
  add_definitions(-Wall)
endif ( USE_DEBUG )

set(LINK_FLAGS " ")

if (USE_CUDA)
    find_package(CUDA)
    # add_definitions(-DUSE_CUDA)
endif (USE_CUDA)

# check if we're using Intel's compiler
if ( CMAKE_CXX_COMPILER MATCHES ".*icpc$" )
    add_definitions(-wd981 -wd383 -wd444 -wd1224 -wd1572)
    if ( NOT USE_DEBUG )
        set(CMAKE_CXX_FLAGS_RELEASE "-ipo -O3 -no-prec-div -xHost -DNDEBUG")
    endif ( NOT USE_DEBUG )
else ( CMAKE_CXX_COMPILER MATCHES ".*icpc$" )
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif ( CMAKE_CXX_COMPILER MATCHES ".*icpc$" )

add_subdirectory(src)
