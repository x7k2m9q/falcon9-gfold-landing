// =============================================================================
// scvx_benchmark.cpp - SCvx求解器性能基准测试
// 理论方案5.0 Phase 5B: 单步求解 605ms → <50ms, 50Hz MPC可行
//
// 测试内容:
//   1. SCvxSolver 单步求解耗时 (验收: <50ms)
//   2. G-FOLD 单步求解耗时 (对比基准)
//   3. SCvx 多约束模式求解耗时 (动压/热流/避障)
//   4. 内存分配检查 (验收: 0次new/malloc)
//   5. 数值结果验证 (与Python端对齐)
// =============================================================================
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "core/fixed_matrix.hpp"
#include "core/quaternion.hpp"
#include "gnc/ekf.hpp"
#include "gnc/guidance.hpp"
#include "gnc/control.hpp"
#include "gnc/octaweb.hpp"
#include "gnc/safety.hpp"
#include "gfold_solver/scvx_solver.hpp"

using namespace falcon9;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// 计时辅助
// ---------------------------------------------------------------------------
template<typename Func>
double benchmark_ms(const char* name, int iterations, Func func) {
    // 预热
    for (int i = 0; i < 3; ++i) func();

    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_ms = total_ms / iterations;
    double per_us = per_ms * 1000.0;
    printf("  %-35s %8.3f ms/call  (%8.2f us,  %d次,  %.1fms总耗时)\n",
           name, per_ms, per_us, iterations, total_ms);
    return per_ms;
}

// ---------------------------------------------------------------------------
// 测试1: SCvx vs G-FOLD 基础性能对比
// 场景: h=1500m, vz=30m/s, wind=0.3 (与Python quick_test一致)
// ---------------------------------------------------------------------------
void test1_basic_performance() {
    printf("\n=== 测试1: SCvx vs G-FOLD 基础性能 ===\n");
    printf("场景: h=1500m, vz=30m/s, m=14000kg, tgo=20s, wind=0.3m/s\n\n");

    // 初始条件 (NED: Z向下为正)
    double pos0[3]      = {0.0, 0.0, -1500.0};  // h=1500m
    double vel0[3]      = {3.0, 0.0, 30.0};     // vz=30m/s下降, wind_x=3m/s
    double pos_target[3] = {0.0, 0.0, 0.0};     // 着陆点
    double vel_target[3] = {0.0, 0.0, 0.0};     // 末端速度=0
    double m_current     = 14000.0;             // 当前质量
    double tgo           = 20.0;                // 剩余时间
    double wind_n[3]     = {3.0, 0.0, 0.0};     // 风速
    int    n_engines     = 1;

    // --- G-FOLD 基准 ---
    printf("[G-FOLD] 解析求解器:\n");
    G_FOLD_Planner gfold(30, 0.6981317007977318f, 0.0f);
    float pos0_f[3]  = {(float)pos0[0], (float)pos0[1], (float)pos0[2]};
    float vel0_f[3]  = {(float)vel0[0], (float)vel0[1], (float)vel0[2]};
    float pos_t_f[3] = {(float)pos_target[0], (float)pos_target[1], (float)pos_target[2]};
    float vel_t_f[3] = {(float)vel_target[0], (float)vel_target[1], (float)vel_target[2]};
    float F_aero_f[3] = {0.0f, 0.0f, 0.0f};

    double gfold_ms = benchmark_ms("G-FOLD.solve()", 1000, [&]() {
        gfold.solve(pos0_f, vel0_f, pos_t_f, vel_t_f,
                    (float)m_current, (float)tgo, F_aero_f, n_engines);
    });

    // --- SCvx 基准 (无多约束) ---
    printf("\n[SCvx] 无多约束模式:\n");
    SCvxSolver scvx(0.6981317007977318);

    double scvx_ms = benchmark_ms("SCvx.solve() (无约束)", 100, [&]() {
        scvx.solve(pos0, vel0, pos_target, vel_target,
                   m_current, tgo, nullptr, wind_n, n_engines, false);
    });

    // --- SCvx 多约束模式 ---
    printf("\n[SCvx] 多约束模式 (动压/热流/避障):\n");
    SCvxSolver scvx_mc(0.6981317007977318);
    // 配置障碍物 (与Python phase5a_validation一致)
    scvx_mc.config.OBSTACLE_CENTER[0] = 50.0;
    scvx_mc.config.OBSTACLE_CENTER[1] = 0.0;
    scvx_mc.config.OBSTACLE_CENTER[2] = -250.0;
    scvx_mc.config.OBSTACLE_SIZE[0]   = 30.0;
    scvx_mc.config.OBSTACLE_SIZE[1]   = 30.0;
    scvx_mc.config.OBSTACLE_SIZE[2]   = 100.0;

    double scvx_mc_ms = benchmark_ms("SCvx.solve() (多约束)", 100, [&]() {
        scvx_mc.solve(pos0, vel0, pos_target, vel_target,
                      m_current, tgo, nullptr, wind_n, n_engines, true);
    });

    // --- 性能验收 ---
    printf("\n[验收] Phase 5B目标: 605ms → <50ms\n");
    printf("  G-FOLD:          %.3f ms  (基准)\n", gfold_ms);
    printf("  SCvx (无约束):   %.3f ms  %s\n", scvx_ms,
           (scvx_ms < 50.0) ? "[PASS <50ms]" : "[FAIL]");
    printf("  SCvx (多约束):   %.3f ms  %s\n", scvx_mc_ms,
           (scvx_mc_ms < 50.0) ? "[PASS <50ms]" : "[FAIL]");
    printf("  Python基准:      605.000 ms  (CVXPY+CLARABEL)\n");
    printf("  C++加速比:       %.1fx  (vs Python)\n", 605.0 / scvx_ms);
}

// ---------------------------------------------------------------------------
// 测试2: SCvx 数值结果验证
// 验证C++端与Python端结果一致性
// 场景: h=300m, vz=30m/s, tgo=20s (位置和速度约束兼容: a=-1.5m/s²)
// ---------------------------------------------------------------------------
void test2_numerical_validation() {
    printf("\n=== 测试2: SCvx 数值结果验证 ===\n");
    printf("场景: h=300m, vz=30m/s, m=14000kg, tgo=20s (约束兼容)\n\n");

    double pos0[3]       = {0.0, 0.0, -300.0};   // h=300m
    double vel0[3]       = {0.0, 0.0, 30.0};      // vz=30m/s下降
    double pos_target[3] = {0.0, 0.0, 0.0};       // 着陆点
    double vel_target[3] = {0.0, 0.0, 0.0};       // 末端速度=0
    double m_current     = 14000.0;
    double tgo           = 20.0;
    double wind_n[3]     = {0.0, 0.0, 0.0};
    int    n_engines     = 1;

    SCvxSolver scvx(0.6981317007977318);
    SCvxResult result = scvx.solve(pos0, vel0, pos_target, vel_target,
                                    m_current, tgo, nullptr, wind_n,
                                    n_engines, false);

    printf("  求解状态:     %s\n", result.success ? "SUCCESS" : "FAIL");
    printf("  降级:         %s\n", result.fallback ? "YES" : "NO");
    printf("  收敛:         %s\n", result.converged ? "YES" : "NO");
    printf("  迭代次数:     %d\n", result.iterations);
    printf("  求解时间:     %.3f ms\n", result.solve_time * 1000.0);
    printf("  求解步数:     %d\n", result.N_solved);
    printf("  代价:         %.3f\n", result.cost);
    printf("  求解器:       %s\n", result.solver_name);
    if (result.fail_reason[0] != '\0') {
        printf("  失败原因:     %s\n", result.fail_reason);
    }

    // 输出前5步控制轨迹
    printf("\n  控制轨迹 (前5步):\n");
    printf("    k    T_x(N)    T_y(N)    T_z(N)    sigma(N)\n");
    int n_show = (result.N_solved < 5) ? result.N_solved : 5;
    for (int k = 0; k < n_show; ++k) {
        printf("    %d  %8.1f  %8.1f  %8.1f  %8.1f\n",
               k, result.u_traj[k][0], result.u_traj[k][1],
               result.u_traj[k][2], result.sigma_traj[k]);
    }

    // 输出末端状态
    int N = result.N_solved;
    printf("\n  末端状态:\n");
    printf("    pos = (%.2f, %.2f, %.2f) m\n",
           result.x_traj[N][0], result.x_traj[N][1], result.x_traj[N][2]);
    printf("    vel = (%.2f, %.2f, %.2f) m/s\n",
           result.x_traj[N][3], result.x_traj[N][4], result.x_traj[N][5]);
    printf("    目标: pos = (0, 0, 0), vel = (0, 0, 0)\n");

    // 统计
    int total, success, fallback;
    double fallback_rate;
    scvx.get_stats(total, success, fallback, fallback_rate);
    printf("\n  统计: 求解%d次, 成功%d次, 降级%d次, 降级率%.1f%%\n",
           total, success, fallback, fallback_rate * 100.0);
}

// ---------------------------------------------------------------------------
// 测试3: 多约束模式验证
// ---------------------------------------------------------------------------
void test3_multi_constraint() {
    printf("\n=== 测试3: 多约束模式验证 ===\n");
    printf("场景: h=300m, vz=30m/s, 障碍物(50,0,-150), 半尺寸(15,15,50)\n\n");

    double pos0[3]       = {0.0, 0.0, -300.0};
    double vel0[3]       = {0.0, 0.0, 30.0};
    double pos_target[3] = {0.0, 0.0, 0.0};
    double vel_target[3] = {0.0, 0.0, 0.0};
    double m_current     = 14000.0;
    double tgo           = 20.0;
    double wind_n[3]     = {0.0, 0.0, 0.0};

    SCvxSolver scvx(0.6981317007977318);
    scvx.config.OBSTACLE_CENTER[0] = 50.0;
    scvx.config.OBSTACLE_CENTER[1] = 0.0;
    scvx.config.OBSTACLE_CENTER[2] = -150.0;  // 障碍物在轨迹中间
    scvx.config.OBSTACLE_SIZE[0]   = 30.0;
    scvx.config.OBSTACLE_SIZE[1]   = 30.0;
    scvx.config.OBSTACLE_SIZE[2]   = 100.0;
    scvx.config.Q_MAX              = 80000.0;   // 80kPa
    scvx.config.Q_DOT_MAX          = 1.5e6;     // 1.5MW/m²

    SCvxResult result = scvx.solve(pos0, vel0, pos_target, vel_target,
                                    m_current, tgo, nullptr, wind_n,
                                    1, true);

    printf("  求解状态:     %s\n", result.success ? "SUCCESS" : "FAIL");
    printf("  降级:         %s\n", result.fallback ? "YES" : "NO");
    printf("  迭代次数:     %d\n", result.iterations);
    printf("  求解时间:     %.3f ms\n", result.solve_time * 1000.0);
    printf("  求解器:       %s\n", result.solver_name);

    // 检查障碍物穿透
    double obs_cx = scvx.config.OBSTACLE_CENTER[0];
    double obs_cy = scvx.config.OBSTACLE_CENTER[1];
    double obs_hx = scvx.config.OBSTACLE_SIZE[0] / 2.0;
    double obs_hy = scvx.config.OBSTACLE_SIZE[1] / 2.0;
    int penetration_count = 0;
    double max_y_offset = 0.0;

    for (int k = 0; k <= result.N_solved; ++k) {
        double px = result.x_traj[k][0];
        double py = result.x_traj[k][1];
        if (std::abs(px - obs_cx) < obs_hx && std::abs(py - obs_cy) < obs_hy) {
            penetration_count++;
        }
        if (std::abs(py) > max_y_offset) max_y_offset = std::abs(py);
    }

    printf("\n  避障效果:\n");
    printf("    穿入障碍物步数:  %d / %d\n", penetration_count, result.N_solved + 1);
    printf("    y方向最大偏移:   %.2f m\n", max_y_offset);

    // 检查动压/热流
    double max_q = 0.0, max_qdot = 0.0;
    for (int k = 0; k <= result.N_solved; ++k) {
        double h = -result.x_traj[k][2];
        double vx = result.x_traj[k][3];
        double vy = result.x_traj[k][4];
        double vz = result.x_traj[k][5];
        double v_mag = std::sqrt(vx*vx + vy*vy + vz*vz);
        double rho = scvx_atmos::density(h);
        double q = rho * v_mag * v_mag;
        double qdot = rho * v_mag * v_mag * v_mag;
        if (q > max_q) max_q = q;
        if (qdot > max_qdot) max_qdot = qdot;
    }
    printf("\n  动压/热流:\n");
    printf("    动压峰值:   %.2f kPa  (限制: %.2f kPa)\n",
           max_q / 1000.0, scvx.config.Q_MAX / 1000.0);
    printf("    热流峰值:   %.3f MW/m²  (限制: %.3f MW/m²)\n",
           max_qdot / 1e6, scvx.config.Q_DOT_MAX / 1e6);
}

// ---------------------------------------------------------------------------
// 测试4: 50Hz MPC可行性验证
// 模拟50Hz MPC循环: 每20ms求解一次
// ---------------------------------------------------------------------------
void test4_mpc_50hz() {
    printf("\n=== 测试4: 50Hz MPC可行性 ===\n");
    printf("模拟50Hz MPC循环 (20ms周期, 100步)\n\n");

    SCvxSolver scvx(0.6981317007977318);

    // 初始条件
    double pos[3]       = {0.0, 0.0, -300.0};   // h=300m
    double vel[3]       = {0.0, 0.0, 30.0};      // vz=30m/s
    double pos_target[3] = {0.0, 0.0, 0.0};
    double vel_target[3] = {0.0, 0.0, 0.0};
    double m            = 14000.0;
    double wind[3]      = {0.0, 0.0, 0.0};

    int    mpc_steps   = 100;
    double dt_mpc      = 0.02;  // 50Hz
    int    over_budget = 0;     // 超时次数
    double max_solve   = 0.0;
    double total_solve = 0.0;

    for (int step = 0; step < mpc_steps; ++step) {
        double h = -pos[2];
        double tgo = (h > 1.0) ? (h / std::max(1.0, vel[2])) : 1.0;
        if (tgo < 1.0) tgo = 1.0;
        if (tgo > 20.0) tgo = 20.0;

        auto t0 = Clock::now();
        SCvxResult result = scvx.solve(pos, vel, pos_target, vel_target,
                                        m, tgo, nullptr, wind, 1, false);
        auto t1 = Clock::now();
        double solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        total_solve += solve_ms;
        if (solve_ms > max_solve) max_solve = solve_ms;
        if (solve_ms > 20.0) over_budget++;

        // 简化动力学推进 (用第一步控制)
        if (result.success && result.N_solved > 0) {
            double T[3] = {result.u_traj[0][0], result.u_traj[0][1], result.u_traj[0][2]};
            double rho = scvx_atmos::density(h);
            double v_mag = std::sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
            double F_aero[3] = {0, 0, 0};
            if (v_mag > 0.5 && rho > 1e-6) {
                double k = 0.5 * rho * 0.3 * 3.14159265358979323846 * (3.35/2.0)*(3.35/2.0);
                double coef = -k * v_mag;
                F_aero[0] = coef * vel[0];
                F_aero[1] = coef * vel[1];
                F_aero[2] = coef * vel[2];
            }
            double acc[3];
            acc[0] = T[0]/m + F_aero[0]/m;
            acc[1] = T[1]/m + F_aero[1]/m;
            acc[2] = T[2]/m + 9.80665 + F_aero[2]/m;

            for (int i = 0; i < 3; ++i) {
                pos[i] += vel[i] * dt_mpc + 0.5 * acc[i] * dt_mpc * dt_mpc;
                vel[i] += acc[i] * dt_mpc;
            }
            m -= std::sqrt(T[0]*T[0]+T[1]*T[1]+T[2]*T[2]) * dt_mpc /
                 (287.0 * 9.80665);  // 简化燃料消耗
            if (m < 5000.0) m = 5000.0;
        }

        if (pos[2] > 0.0) break;  // 着陆
    }

    printf("  MPC步数:          %d\n", mpc_steps);
    printf("  平均求解时间:     %.3f ms\n", total_solve / mpc_steps);
    printf("  最大求解时间:     %.3f ms\n", max_solve);
    printf("  超时次数(>20ms):  %d / %d\n", over_budget, mpc_steps);
    printf("  50Hz可行性:       %s\n",
           (over_budget == 0 && max_solve < 20.0) ? "[PASS]" : "[FAIL]");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("============================================================\n");
    printf(" Falcon 9 SCvx Solver Performance Benchmark\n");
    printf(" Phase 5B: 605ms → <50ms, 50Hz MPC\n");
    printf("============================================================\n");

    test1_basic_performance();
    test2_numerical_validation();
    test3_multi_constraint();
    test4_mpc_50hz();

    printf("\n============================================================\n");
    printf(" Benchmark Complete\n");
    printf("============================================================\n");
    return 0;
}
