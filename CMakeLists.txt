cmake_minimum_required(VERSION 3.18.2 FATAL_ERROR)
project(C6NNUE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")


#==========================================================
# CMake Options

option(USE_SSE  "Enable SSE2/SSSE3/SSE4.1 instruction" ON)
option(USE_AVX2 "Enable AVX2/FMA instruction" ON)
option(USE_AVX512 "Enable AVX512 instruction" OFF)
option(USE_VNNI "Enable AVX512_VNNI/AVX_VNNI instruction" OFF)
option(USE_NEON "Enable NEON instruction" OFF)
option(USE_NEON_DOTPROD "Enable NEON dotprod instruction" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "No build type selected, default to Release" FORCE)
endif()

#==========================================================

# 定义源文件
set(SOURCES
        mix9nnue_test.cpp
        eval/evaluator.h
        eval/mix9nnue.cpp
        eval/mix9nnue.h
        eval/simdops.h
        eval/weightloader.h
        game/game.h
        game/game.cpp
        search/node.h
        search/node.cpp
        search/mcts.h
        search/mcts.cpp
)

# 创建可执行文件
add_executable(C6NNUE ${SOURCES})
set_target_properties(C6NNUE PROPERTIES OUTPUT_NAME "c6nnue")

# 定义编译标志
target_compile_definitions(C6NNUE PRIVATE ${ARCH_DEFINITIONS})

#==========================================================
# Compile flags

if(USE_SSE)
    target_compile_definitions(C6NNUE PRIVATE USE_SSE)
endif()
if(USE_AVX2)
    target_compile_definitions(C6NNUE PRIVATE USE_AVX2)
endif()
if(USE_AVX512)
    target_compile_definitions(C6NNUE PRIVATE USE_AVX512)
endif()
if(USE_VNNI)
    target_compile_definitions(C6NNUE PRIVATE USE_VNNI)
endif()
if(USE_NEON OR USE_NEON_DOTPROD)
    target_compile_definitions(C6NNUE PRIVATE USE_NEON)
endif()
if(USE_NEON_DOTPROD)
    target_compile_definitions(C6NNUE PRIVATE USE_NEON_DOTPROD)
endif()

if(MSVC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "Setting up build for MSVC Clang-CL.")
        string(REGEX REPLACE "/O2" "/clang:-O3" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    else()
        message(STATUS "Setting up build for MSVC.")
        string(REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    endif()

    if(USE_AVX512)
        # Fix SSE3, SSSE3, SSE4.1, AVX, AVX2, FMA, AVX512F, AVX512DQ, AVX512BW flags on MSVC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE3__ /D__SSSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /D__FMA__ /D__AVX512F__ /D__AVX512DQ__ /D__AVX512BW__")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavx512f /clang:-mavx512dq /clang:-mavx512bw")
            if(USE_VNNI)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__AVX512VNNI__")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavx512vnni")
            endif()
        endif()
    elseif(USE_AVX2)
        # Fix SSE3, SSSE3, SSE4.1, AVX, AVX2, FMA flags on MSVC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE3__ /D__SSSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /D__FMA__")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavx2 /clang:-mfma")
            if(USE_VNNI)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavxvnni")
            endif()
        endif()
    elseif(USE_SSE)
        # Fix SSE3, SSSE3, SSE4.1 flags on MSVC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE3__ /D__SSSE3__ /D__SSE4_1__")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-msse /clang:-msse2 /clang:-msse3 /clang:-mssse3 /clang:-msse4 /clang:-msse4.1")
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(USE_NEON_DOTPROD)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-march=armv8.2-a+dotprod")
        elseif(USE_NEON)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-march=armv8-a+simd")
        endif()
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
        set_target_properties(C6NNUE PROPERTIES LINK_FLAGS "/PROFILE")
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL Release)
        set_property(TARGET C6NNUE PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Setting up build for GNU or Clang.")

    if(USE_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512bw")
        if(USE_VNNI)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vnni -mavx512vl")
        endif()
    elseif(USE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        if(USE_VNNI)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavxvnni")
        endif()
    elseif(USE_SSE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1")
    endif()

    if(USE_NEON_DOTPROD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod")
    elseif(USE_NEON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    endif()

    if(WIN32)
        target_link_libraries(C6NNUE PRIVATE pthread)
    endif()
endif()

