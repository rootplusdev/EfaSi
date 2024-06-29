cmake_minimum_required(VERSION 3.18.2 FATAL_ERROR)
project(C6NNUE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 检测系统架构并设置编译标志
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 针对ARM架构
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    set(ARCH_DEFINITIONS USE_NEON USE_NEON_DOTPROD)
    message("ARM.")
else()
    # 针对x86架构
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    set(ARCH_DEFINITIONS USE_SSE USE_AVX2)
    message("X86.")
endif()

# 设置Release模式的优化标志
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

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

# 定义编译标志
target_compile_definitions(C6NNUE PRIVATE ${ARCH_DEFINITIONS})
