cmake_minimum_required(VERSION 4.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 23)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# --------------------------------------------------------------
# Project Configuration
# --------------------------------------------------------------

file(READ "Include/Shark/Core/Detail/Setup.hpp" SKT_SETUP_FILE)
string(REGEX MATCH "#define[ ]+SKT_VERSION_MAJOR[ ]+([0-9]+)" _ ${SKT_SETUP_FILE})
string(REGEX MATCH "#define[ ]+SKT_VERSION_MINOR[ ]+([0-9]+)" _ ${SKT_SETUP_FILE})
string(REGEX MATCH "#define[ ]+SKT_VERSION_PATCH[ ]+([0-9]+)" _ ${SKT_SETUP_FILE})
set(SKT_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(SKT_VERSION_MINOR "${CMAKE_MATCH_1}")
set(SKT_VERSION_PATCH "${CMAKE_MATCH_1}")
set(SKT_VERSION ${SKT_VERSION_MAJOR}.${SKT_VERSION_MINOR}.${SKT_VERSION_PATCH})

project(Shark
        LANGUAGES CXX C
        VERSION ${SKT_VERSION})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# --------------------------------------------------------------
# Global settings
# --------------------------------------------------------------

if (CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;MinSizeRel;RelWithDebInfo" CACHE STRING "" FORCE)
else ()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;MinSizeRel;RelWithDebInfo")
endif ()

if (PROJECT_IS_TOP_LEVEL)
    include(CPM)
endif ()

set(BEE_BUILD_TESTS PROJECT_IS_TOP_LEVEL CACHE BOOL "Build test programs")
set(BEE_BUILD_BENCHMARKS PROJECT_IS_TOP_LEVEL CACHE BOOL "Build benchmark programs")
set(BEE_BUILD_SAMPLES PROJECT_IS_TOP_LEVEL CACHE BOOL "Build sample programs")

include(HelperScripts)

# --------------------------------------------------------------
# Sources
# --------------------------------------------------------------

file(GLOB_RECURSE Sources
        "${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cpp"
)

file(GLOB_RECURSE Headers
        "${CMAKE_CURRENT_SOURCE_DIR}/Include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/Include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Include/*.inl"
)

add_library(Shark STATIC ${Sources} ${Headers})

target_include_directories(Shark
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:Include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Include/Shark
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

SetCompilerFlags(Shark)

target_compile_definitions(Shark
        PUBLIC
        $<$<CONFIG:Release>:NDEBUG>
        $<$<CONFIG:Debug>:_DEBUG>
        # Windows
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
        $<$<PLATFORM_ID:Windows>:UNICODE>
        # MSVC C++ 
        $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
        $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>
        $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
        $<$<CXX_COMPILER_ID:MSVC>:_ENABLE_EXTENDED_ALIGNED_STORAGE>
        $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING>
        PRIVATE
        # Clang.
        $<$<CXX_COMPILER_ID:Clang>:_MSC_EXTENSIONS> # enable MS extensions
        # $<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>
)

if (MSVC)
    target_compile_options(Shark PUBLIC /utf-8)
else ()
    target_compile_options(Shark PUBLIC -finput-charset=UTF-8 -fexec-charset=UTF-8)
endif ()

add_library(Shark::Shark ALIAS Shark)

# Samples
if (BEE_BUILD_SAMPLES)
    add_subdirectory(Samples)
endif ()

# Tests
add_subdirectory(Tests)