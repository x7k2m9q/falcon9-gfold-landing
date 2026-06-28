# =============================================================================
# CMakeLists.txt - 猎鹰9号 G-FOLD 着陆 C++ 飞控仿真
# 理论方案3.0 Phase 2-3: 嵌入式C++翻译 + FreeRTOS软硬件协同验证
#
# 构建目标:
#   falcon9_sim   - 主程序 (UDP + FreeRTOS任务模拟)
#   falcon9_test  - 单元测试 (7个模块)
#   falcon9_bench - 性能基准 (ARM Cortex-R5F @ 600MHz 验收)
#
# 构建:
#   cmake -B build
#   cmake --build build
#
# 约法三章:
#   1. 零动态内存 (无 vector/string/new/malloc)
#   2. float32 默认, double 仅用于 EKF 协方差 P
#   3. 无 OSQP/CVXPY 依赖 (纯解析求解)
# =============================================================================
cmake_minimum_required(VERSION 3.10)
project(falcon9_gfold LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项: 严格警告, 优化
if(MSVC)
    add_compile_options(/W4 /O2 /EHsc /utf-8)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DWIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR})

# =============================================================================
# 目标1: falcon9_sim - 主仿真程序
# =============================================================================
add_executable(falcon9_sim
    main.cpp
)
target_link_libraries(falcon9_sim PRIVATE)

# Windows 需要链接 ws2_32 (Winsock) 和 pthread
if(WIN32)
    target_link_libraries(falcon9_sim PRIVATE ws2_32)
    if(MSVC)
        target_link_libraries(falcon9_sim PRIVATE)
    else()
        target_link_libraries(falcon9_sim PRIVATE pthread)
    endif()
else()
    target_link_libraries(falcon9_sim PRIVATE pthread)
endif()

# =============================================================================
# 目标2: falcon9_test - 单元测试
# =============================================================================
add_executable(falcon9_test
    tests/test_golden_data.cpp
)
if(WIN32)
    target_link_libraries(falcon9_test PRIVATE ws2_32)
    if(NOT MSVC)
        target_link_libraries(falcon9_test PRIVATE pthread)
    endif()
else()
    target_link_libraries(falcon9_test PRIVATE pthread)
endif()

# =============================================================================
# 目标3: falcon9_bench - 性能基准
# =============================================================================
add_executable(falcon9_bench
    tests/benchmark.cpp
)
if(WIN32)
    target_link_libraries(falcon9_bench PRIVATE ws2_32)
    if(NOT MSVC)
        target_link_libraries(falcon9_bench PRIVATE pthread)
    endif()
else()
    target_link_libraries(falcon9_bench PRIVATE pthread)
endif()

# =============================================================================
# 目标4: falcon9_align - Golden Data 单帧对齐测试
# =============================================================================
add_executable(falcon9_align
    tests/golden_align.cpp
)
if(WIN32)
    target_link_libraries(falcon9_align PRIVATE ws2_32)
    if(NOT MSVC)
        target_link_libraries(falcon9_align PRIVATE pthread)
    endif()
else()
    target_link_libraries(falcon9_align PRIVATE pthread)
endif()

# =============================================================================
# 目标5: falcon9_scvx_bench - SCvx求解器性能基准 (Phase 5B)
# 验收: 单步求解 605ms → <50ms, 50Hz MPC可行
# =============================================================================
add_executable(falcon9_scvx_bench
    tests/scvx_benchmark.cpp
)
if(WIN32)
    target_link_libraries(falcon9_scvx_bench PRIVATE ws2_32)
    if(NOT MSVC)
        target_link_libraries(falcon9_scvx_bench PRIVATE pthread)
    endif()
else()
    target_link_libraries(falcon9_scvx_bench PRIVATE pthread)
endif()

# =============================================================================
# 安装规则 (可选)
# =============================================================================
install(TARGETS falcon9_sim falcon9_test falcon9_bench falcon9_scvx_bench
        RUNTIME DESTINATION bin)

# =============================================================================
# 信息输出
# =============================================================================
message(STATUS "========================================")
message(STATUS " Falcon 9 G-FOLD Landing C++ Simulation")
message(STATUS " Phase 2-3: Embedded C++ + FreeRTOS")
message(STATUS "========================================")
message(STATUS " C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS " Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " Targets: falcon9_sim, falcon9_test, falcon9_bench")
message(STATUS "========================================")
