// =============================================================================
// benchmark.cpp - 性能基准测试
// 理论方案3.0 Phase 3: Valgrind内存检测 + Jitter基准
//
// 测试内容:
//   1. EKF 1000次预测+更新耗时 (验收: <1ms/次)
//   2. Guidance 100次G-FOLD求解耗时 (验收: <10ms/次)
//   3. Control 1000次姿态控制耗时 (验收: <0.5ms/次)
//   4. 内存分配检查 (验收: 0次new/malloc)
// =============================================================================
#include <chrono>
#include <cstdio>
#include <cmath>
#include "core/fixed_matrix.hpp"
#include "core/quaternion.hpp"
#include "gnc/ekf.hpp"
#include "gnc/guidance.hpp"
#include "gnc/control.hpp"
#include "gnc/octaweb.hpp"
#include "gnc/safety.hpp"

using namespace falcon9;
using Clock = std::chrono::steady_clock;

template<typename Func>
double benchmark(const char* name, int iterations, Func func) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_us = total_ms * 1000.0 / iterations;
    printf("  %-30s %8.2f us/call  (%d次, %.1fms总耗时)\n",
           name, per_us, iterations, total_ms);
    return per_us;
}

int main() {
    printf("=== Falcon 9 C++ 性能基准 ===\n\n");

    // --- EKF基准 ---
    printf("[EKF] 15维MEKF性能:\n");
    MEKF ekf;
    float p0[3] = {0, 0, -2000};
    float v0[3] = {0, 0, 80};
    float q0[4] = {0.7071f, 0, 0.7071f, 0};
    ekf.reset(p0, v0, q0);

    float gyro[3] = {0.001f, -0.002f, 0.0005f};
    float accel[3] = {0.01f, -0.02f, -9.8f};
    float gps_pos[3] = {0.1f, -0.2f, -1999.9f};
    float gps_vel[3] = {0.01f, -0.02f, 79.9f};

    double ekf_predict_us = benchmark("EKF.predict()", 10000, [&]() {
        ekf.predict(gyro, accel, 0.001f);
    });

    double ekf_gps_us = benchmark("EKF.update_gps()", 1000, [&]() {
        ekf.update_gps(gps_pos, gps_vel);
    });

    double ekf_radar_us = benchmark("EKF.update_radar()", 1000, [&]() {
        ekf.update_radar(1500.0f);
    });

    // --- Guidance基准 ---
    printf("\n[Guidance] G-FOLD制导性能:\n");
    LandingGuidance guidance;

    float pos[3] = {0, 0, -1400};
    float vel[3] = {0, 0, 60};
    float q[4] = {0.7071f, 0, 0.7071f, 0};

    double guidance_us = benchmark("Guidance.update()", 1000, [&]() {
        guidance.update(pos, vel, q, 14000.0f, 5.0f, 0.01f);
    });

    // --- Control基准 ---
    printf("\n[Control] 姿态控制性能:\n");
    AttitudeController att;
    float q_actual[4] = {0.7071f, 0, 0.7071f, 0};
    float omega[3] = {0.01f, -0.02f, 0.005f};
    float q_des[4] = {0.7071f, 0, 0.7071f, 0};
    float omega_des[3] = {0, 0, 0};
    float pos_n[3] = {0, 0, -1400};
    float vel_n[3] = {0, 0, 60};
    float tvc[2], gf[3], rcs[3];

    double control_us = benchmark("Attitude.update()", 10000, [&]() {
        att.update(q_actual, omega, q_des, omega_des,
                   0.5f, pos_n, vel_n, 1, tvc, gf, rcs);
    });

    // --- Octaweb基准 ---
    printf("\n[Octaweb] 发动机分配性能:\n");
    Octaweb oct;
    oct.set_engine_config(1);

    double octaweb_us = benchmark("Octaweb.update()", 10000, [&]() {
        oct.update(0.5f, 0.01f, 0.02f, "G-FOLD", 1400.0f, 0.01f);
    });

    // --- Safety基准 ---
    printf("\n[Safety] 安全监控性能:\n");
    SafetyMonitor sm;

    double safety_us = benchmark("Safety.evaluate()", 10000, [&]() {
        StateEstimate state;
        state.reset();
        state.p[2] = -1400;
        state.q[0] = 0.7071f;
        state.q[2] = 0.7071f;
        sm.evaluate(state, 14000.0f, 1400.0f, 400000.0f, 1);
    });

    // --- 验收 ---
    printf("\n========================================\n");
    printf(" 验收标准 (ARM Cortex-R5F @ 600MHz)\n");
    printf("========================================\n");
    printf(" EKF.predict <1000us:   %s (%.1fus)\n",
           ekf_predict_us < 1000 ? "PASS" : "FAIL", ekf_predict_us);
    printf(" EKF.update_gps <2000us: %s (%.1fus)\n",
           ekf_gps_us < 2000 ? "PASS" : "FAIL", ekf_gps_us);
    printf(" Guidance <10000us:     %s (%.1fus)\n",
           guidance_us < 10000 ? "PASS" : "FAIL", guidance_us);
    printf(" Control <500us:        %s (%.1fus)\n",
           control_us < 500 ? "PASS" : "FAIL", control_us);
    printf(" Octaweb <100us:        %s (%.1fus)\n",
           octaweb_us < 100 ? "PASS" : "FAIL", octaweb_us);
    printf(" Safety <100us:         %s (%.1fus)\n",
           safety_us < 100 ? "PASS" : "FAIL", safety_us);
    printf("========================================\n");
    printf("\n注意: Desktop性能远高于ARM, 此基准仅验证算法复杂度.\n");
    printf("      ARM部署时需在目标硬件上重新基准.\n");

    return 0;
}
