// =============================================================================
// golden_align.cpp - Golden Data 单帧对齐测试 (C++ 端)
//
// 用户强制要求: 停止跑联合仿真, 用开环单帧对比定位 bug.
// 读取 Python 导出的冻结状态 (golden_data/frozen_t10.txt),
// 设置 FlightComputer 内部状态, 执行单步 process_control(),
// 打印所有中间变量, 供与 Python 端逐层对比.
//
// 用法: falcon9_align [frozen_file]
//   frozen_file 默认 ../golden_data/frozen_t10.txt
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "core/fixed_matrix.hpp"
#include "core/quaternion.hpp"
#include "core/ring_buffer.hpp"
#include "core/types.hpp"
#include "gnc/ekf.hpp"
#include "gnc/guidance.hpp"
#include "gnc/control.hpp"
#include "gnc/octaweb.hpp"
#include "gnc/safety.hpp"
#include "os/freertos_sim.hpp"

using namespace falcon9;

// ===========================================================================
// 冻结状态结构
// ===========================================================================
struct FrozenState {
    double t, fuel, mass, cg_x, h, T_max_single;
    double I_body[3];
    double ekf_p[3], ekf_v[3], ekf_q[4], omega[3];
    double M_aero_prev[3];
    char   phase[32];
    int    solve_counter, solve_period;
    double last_solve_time;
    double q_des_prev[4], q_des_current[4], q_des_target[4];
    double throttle_prev, throttle_current, throttle_target;
    int    bang_mode;
    double bang_last_throttle;
    double F_aero_n_prev[3];
    int    n_engines_current;
    double gfold_t0, gfold_tgo_fixed;
    int    octaweb_n_active;
    double octaweb_engines0_thrust;
    // Python 参考输出
    double ref_throttle;
    double ref_q_des[4];
    int    ref_n_engines;
    double ref_M_cmd[3];
    double ref_tvc_gimbal[2];
    double ref_total_thrust;
    double ref_gfold_u0[3];
    double ref_gfold_T_mag;
    double ref_gfold_tgo;
    bool   has_ref_gfold;
};

// ===========================================================================
// 解析冻结状态文件
// ===========================================================================
bool parse_frozen_file(const char* filepath, FrozenState& fs) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        printf("[ERROR] 无法打开冻结状态文件: %s\n", filepath);
        return false;
    }

    // 初始化默认值
    memset(&fs, 0, sizeof(fs));
    fs.has_ref_gfold = false;
    fs.solve_period = 100;

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        // 跳过注释和空行
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        char key[64];
        if (sscanf(line, "%63s", key) != 1) continue;

        if (strcmp(key, "t") == 0) {
            sscanf(line, "%*s %lf", &fs.t);
        } else if (strcmp(key, "fuel") == 0) {
            sscanf(line, "%*s %lf", &fs.fuel);
        } else if (strcmp(key, "mass") == 0) {
            sscanf(line, "%*s %lf", &fs.mass);
        } else if (strcmp(key, "cg_x") == 0) {
            sscanf(line, "%*s %lf", &fs.cg_x);
        } else if (strcmp(key, "I_body") == 0) {
            sscanf(line, "%*s %lf %lf %lf",
                   &fs.I_body[0], &fs.I_body[1], &fs.I_body[2]);
        } else if (strcmp(key, "h") == 0) {
            sscanf(line, "%*s %lf", &fs.h);
        } else if (strcmp(key, "T_max_single") == 0) {
            sscanf(line, "%*s %lf", &fs.T_max_single);
        } else if (strcmp(key, "ekf_p") == 0) {
            sscanf(line, "%*s %lf %lf %lf",
                   &fs.ekf_p[0], &fs.ekf_p[1], &fs.ekf_p[2]);
        } else if (strcmp(key, "ekf_v") == 0) {
            sscanf(line, "%*s %lf %lf %lf",
                   &fs.ekf_v[0], &fs.ekf_v[1], &fs.ekf_v[2]);
        } else if (strcmp(key, "ekf_q") == 0) {
            sscanf(line, "%*s %lf %lf %lf %lf",
                   &fs.ekf_q[0], &fs.ekf_q[1], &fs.ekf_q[2], &fs.ekf_q[3]);
        } else if (strcmp(key, "omega") == 0) {
            sscanf(line, "%*s %lf %lf %lf",
                   &fs.omega[0], &fs.omega[1], &fs.omega[2]);
        } else if (strcmp(key, "M_aero_prev") == 0) {
            sscanf(line, "%*s %lf %lf %lf",
                   &fs.M_aero_prev[0], &fs.M_aero_prev[1], &fs.M_aero_prev[2]);
        } else if (strcmp(key, "phase") == 0) {
            sscanf(line, "%*s %31s", fs.phase);
        } else if (strcmp(key, "solve_counter") == 0) {
            sscanf(line, "%*s %d", &fs.solve_counter);
        } else if (strcmp(key, "solve_period") == 0) {
            sscanf(line, "%*s %d", &fs.solve_period);
        } else if (strcmp(key, "last_solve_time") == 0) {
            sscanf(line, "%*s %lf", &fs.last_solve_time);
        } else if (strcmp(key, "q_des_prev") == 0) {
            sscanf(line, "%*s %lf %lf %lf %lf",
                   &fs.q_des_prev[0], &fs.q_des_prev[1],
                   &fs.q_des_prev[2], &fs.q_des_prev[3]);
        } else if (strcmp(key, "q_des_current") == 0) {
            sscanf(line, "%*s %lf %lf %lf %lf",
                   &fs.q_des_current[0], &fs.q_des_current[1],
                   &fs.q_des_current[2], &fs.q_des_current[3]);
        } else if (strcmp(key, "q_des_target") == 0) {
            sscanf(line, "%*s %lf %lf %lf %lf",
                   &fs.q_des_target[0], &fs.q_des_target[1],
                   &fs.q_des_target[2], &fs.q_des_target[3]);
        } else if (strcmp(key, "throttle_prev") == 0) {
            sscanf(line, "%*s %lf", &fs.throttle_prev);
        } else if (strcmp(key, "throttle_current") == 0) {
            sscanf(line, "%*s %lf", &fs.throttle_current);
        } else if (strcmp(key, "throttle_target") == 0) {
            sscanf(line, "%*s %lf", &fs.throttle_target);
        } else if (strcmp(key, "bang_mode") == 0) {
            sscanf(line, "%*s %d", &fs.bang_mode);
        } else if (strcmp(key, "bang_last_throttle") == 0) {
            sscanf(line, "%*s %lf", &fs.bang_last_throttle);
        } else if (strcmp(key, "F_aero_n_prev") == 0) {
            sscanf(line, "%*s %lf %lf %lf",
                   &fs.F_aero_n_prev[0], &fs.F_aero_n_prev[1],
                   &fs.F_aero_n_prev[2]);
        } else if (strcmp(key, "n_engines_current") == 0) {
            sscanf(line, "%*s %d", &fs.n_engines_current);
        } else if (strcmp(key, "gfold_t0") == 0) {
            sscanf(line, "%*s %lf", &fs.gfold_t0);
        } else if (strcmp(key, "gfold_tgo_fixed") == 0) {
            sscanf(line, "%*s %lf", &fs.gfold_tgo_fixed);
        } else if (strcmp(key, "octaweb_n_active") == 0) {
            sscanf(line, "%*s %d", &fs.octaweb_n_active);
        } else if (strcmp(key, "octaweb_engines0_thrust") == 0) {
            sscanf(line, "%*s %lf", &fs.octaweb_engines0_thrust);
        } else if (strcmp(key, "ref_throttle") == 0) {
            sscanf(line, "%*s %lf", &fs.ref_throttle);
        } else if (strcmp(key, "ref_q_des") == 0) {
            sscanf(line, "%*s %lf %lf %lf %lf",
                   &fs.ref_q_des[0], &fs.ref_q_des[1],
                   &fs.ref_q_des[2], &fs.ref_q_des[3]);
        } else if (strcmp(key, "ref_n_engines") == 0) {
            sscanf(line, "%*s %d", &fs.ref_n_engines);
        } else if (strcmp(key, "ref_M_cmd") == 0) {
            sscanf(line, "%*s %lf %lf %lf",
                   &fs.ref_M_cmd[0], &fs.ref_M_cmd[1], &fs.ref_M_cmd[2]);
        } else if (strcmp(key, "ref_tvc_gimbal") == 0) {
            sscanf(line, "%*s %lf %lf",
                   &fs.ref_tvc_gimbal[0], &fs.ref_tvc_gimbal[1]);
        } else if (strcmp(key, "ref_total_thrust") == 0) {
            sscanf(line, "%*s %lf", &fs.ref_total_thrust);
        } else if (strcmp(key, "ref_gfold_u0") == 0) {
            sscanf(line, "%*s %lf %lf %lf",
                   &fs.ref_gfold_u0[0], &fs.ref_gfold_u0[1], &fs.ref_gfold_u0[2]);
            fs.has_ref_gfold = true;
        } else if (strcmp(key, "ref_gfold_T_mag") == 0) {
            sscanf(line, "%*s %lf", &fs.ref_gfold_T_mag);
        } else if (strcmp(key, "ref_gfold_tgo") == 0) {
            sscanf(line, "%*s %lf", &fs.ref_gfold_tgo);
        }
    }

    fclose(f);
    return true;
}

// ===========================================================================
// 设置 FlightComputer 内部状态
// ===========================================================================
void setup_flight_computer(FlightComputer& fc, const FrozenState& fs) {
    // === 1. 设置 latest_state (EKF 输出) ===
    for (int i = 0; i < 3; ++i) {
        fc.latest_state.p[i] = static_cast<float>(fs.ekf_p[i]);
        fc.latest_state.v[i] = static_cast<float>(fs.ekf_v[i]);
        fc.latest_state.omega[i] = static_cast<float>(fs.omega[i]);
    }
    for (int i = 0; i < 4; ++i) {
        fc.latest_state.q[i] = static_cast<float>(fs.ekf_q[i]);
    }
    fc.latest_state.valid = true;

    // === 2. 设置系统状态 ===
    fc.sim_time.store(static_cast<float>(fs.t), std::memory_order_relaxed);
    fc.fuel_mass.store(static_cast<float>(fs.fuel), std::memory_order_relaxed);

    // === 3. 设置 Guidance 内部状态 ===
    // phase
    if (strcmp(fs.phase, "G-FOLD") == 0) {
        fc.guidance.phase = GuidancePhase::GFOLD;
    } else if (strcmp(fs.phase, "DESCENT") == 0) {
        fc.guidance.phase = GuidancePhase::DESCENT;
    } else if (strcmp(fs.phase, "DEADBAND") == 0) {
        fc.guidance.phase = GuidancePhase::DEADBAND;
    } else if (strcmp(fs.phase, "LANDED") == 0) {
        fc.guidance.phase = GuidancePhase::LANDED;
    }

    fc.guidance.solve_counter = fs.solve_counter;
    fc.guidance.solve_period = fs.solve_period;
    fc.guidance.last_solve_time = static_cast<float>(fs.last_solve_time);
    fc.guidance.last_solve_success = true;

    for (int i = 0; i < 4; ++i) {
        fc.guidance.q_des_prev[i] = static_cast<float>(fs.q_des_prev[i]);
        fc.guidance.q_des_current[i] = static_cast<float>(fs.q_des_current[i]);
        fc.guidance.q_des_target[i] = static_cast<float>(fs.q_des_target[i]);
        fc.guidance.q_des[i] = static_cast<float>(fs.q_des_current[i]);
    }

    fc.guidance.throttle_prev = static_cast<float>(fs.throttle_prev);
    fc.guidance.throttle_current = static_cast<float>(fs.throttle_current);
    fc.guidance.throttle_target = static_cast<float>(fs.throttle_target);
    fc.guidance.throttle = static_cast<float>(fs.throttle_current);

    fc.guidance.bang_mode = (fs.bang_mode != 0);
    fc.guidance.bang_last_throttle = static_cast<float>(fs.bang_last_throttle);

    for (int i = 0; i < 3; ++i) {
        fc.guidance.F_aero_n_prev[i] = static_cast<float>(fs.F_aero_n_prev[i]);
    }

    fc.guidance.n_engines_current = fs.n_engines_current;
    fc.guidance.gfold_t0 = static_cast<float>(fs.gfold_t0);
    fc.guidance.gfold_tgo_fixed = static_cast<float>(fs.gfold_tgo_fixed);
    fc.guidance.landed = false;

    // === 4. 设置 Octaweb 状态 ===
    if (fs.octaweb_n_active != fc.octaweb.n_active) {
        fc.octaweb.set_engine_config(fs.octaweb_n_active);
    }
    // 设置中心发动机推力 (用于 TVC 权限计算)
    // 必须同时设置 thrust_ratio, 否则 update() 会用 thrust_ratio*T_max 重算,
    // 把 thrust 重置为 0 (Golden Data 对齐测试发现此 bug).
    float T_max_single_f = static_cast<float>(fs.T_max_single);
    float thrust0 = static_cast<float>(fs.octaweb_engines0_thrust);
    fc.octaweb.engines[0].thrust = thrust0;
    if (T_max_single_f > 1.0f) {
        fc.octaweb.engines[0].thrust_ratio = thrust0 / T_max_single_f;
        fc.octaweb.engines[0].startup_progress = 1.0f;  // 已完成点火
        fc.octaweb.engines[0].is_active = (thrust0 > 1.0f);
    }
    // 外层发动机也需设置 (3发模式)
    for (int i = 1; i < fs.octaweb_n_active && i < 9; ++i) {
        fc.octaweb.engines[i].thrust = thrust0;
        if (T_max_single_f > 1.0f) {
            fc.octaweb.engines[i].thrust_ratio = thrust0 / T_max_single_f;
            fc.octaweb.engines[i].startup_progress = 1.0f;
            fc.octaweb.engines[i].is_active = (thrust0 > 1.0f);
        }
    }
}

// ===========================================================================
// 打印对比结果
// ===========================================================================
void print_comparison(FlightComputer& fc, const FrozenState& fs) {
    printf("\n======================================================================\n");
    printf("Golden Data 对齐测试: t=10.0s 单帧 (C++ 端)\n");
    printf("======================================================================\n");

    // === 1. 冻结状态 (制导输入) ===
    printf("\n--- 1. 冻结状态 (制导输入) ---\n");
    printf("t              = %.4f s\n", fs.t);
    printf("fuel           = %.4f kg\n", fs.fuel);
    printf("mass           = %.4f kg\n", fs.mass);
    printf("h              = %.4f m\n", fs.h);
    printf("T_max_single   = %.4f N\n", fs.T_max_single);
    printf("cg_x           = %.6f m\n", fs.cg_x);
    printf("I_body         = [%.4f, %.4f, %.4f]\n",
           fs.I_body[0], fs.I_body[1], fs.I_body[2]);
    printf("ekf_p (NED)    = [%.6f, %.6f, %.6f]\n",
           fs.ekf_p[0], fs.ekf_p[1], fs.ekf_p[2]);
    printf("ekf_v (NED)    = [%.6f, %.6f, %.6f]\n",
           fs.ekf_v[0], fs.ekf_v[1], fs.ekf_v[2]);
    printf("ekf_q [w,x,y,z]= [%.8f, %.8f, %.8f, %.8f]\n",
           fs.ekf_q[0], fs.ekf_q[1], fs.ekf_q[2], fs.ekf_q[3]);
    printf("omega (b系)    = [%.8f, %.8f, %.8f]\n",
           fs.omega[0], fs.omega[1], fs.omega[2]);
    printf("M_aero_prev    = [%.6f, %.6f, %.6f]\n",
           fs.M_aero_prev[0], fs.M_aero_prev[1], fs.M_aero_prev[2]);

    // === 2. Guidance 内部状态 (求解前) ===
    printf("\n--- 2. Guidance 内部状态 (求解前) ---\n");
    printf("phase          = %s\n", fs.phase);
    printf("solve_counter  = %d\n", fs.solve_counter);
    printf("last_solve_time= %.4f\n", fs.last_solve_time);
    printf("q_des_prev     = [%.8f, %.8f, %.8f, %.8f]\n",
           fs.q_des_prev[0], fs.q_des_prev[1], fs.q_des_prev[2], fs.q_des_prev[3]);
    printf("q_des_current  = [%.8f, %.8f, %.8f, %.8f]\n",
           fs.q_des_current[0], fs.q_des_current[1],
           fs.q_des_current[2], fs.q_des_current[3]);
    printf("q_des_target   = [%.8f, %.8f, %.8f, %.8f]\n",
           fs.q_des_target[0], fs.q_des_target[1],
           fs.q_des_target[2], fs.q_des_target[3]);
    printf("throttle_prev  = %.6f\n", fs.throttle_prev);
    printf("throttle_current= %.6f\n", fs.throttle_current);
    printf("throttle_target= %.6f\n", fs.throttle_target);
    printf("bang_mode      = %d\n", fs.bang_mode);
    printf("bang_last_throttle= %.6f\n", fs.bang_last_throttle);
    printf("F_aero_n_prev  = [%.6f, %.6f, %.6f]\n",
           fs.F_aero_n_prev[0], fs.F_aero_n_prev[1], fs.F_aero_n_prev[2]);
    printf("n_engines_current = %d\n", fs.n_engines_current);

    // === 3. Guidance 输出 (C++) ===
    printf("\n--- 3. Guidance 输出 (C++) ---\n");
    const char* phase_str = "UNKNOWN";
    switch (fc.guidance.phase) {
        case GuidancePhase::DESCENT:  phase_str = "DESCENT";  break;
        case GuidancePhase::GFOLD:    phase_str = "G-FOLD";   break;
        case GuidancePhase::DEADBAND: phase_str = "DEADBAND"; break;
        case GuidancePhase::LANDED:   phase_str = "LANDED";   break;
    }
    printf("phase          = %s\n", phase_str);
    printf("throttle       = %.6f\n", fc.guidance.throttle);
    printf("q_des [w,x,y,z]= [%.8f, %.8f, %.8f, %.8f]\n",
           fc.guidance.q_des[0], fc.guidance.q_des[1],
           fc.guidance.q_des[2], fc.guidance.q_des[3]);
    printf("n_engines      = %d\n", fc.guidance.n_engines_current);
    printf("bang_mode      = %d\n", fc.guidance.bang_mode ? 1 : 0);
    printf("solve_happened = %d\n", fc.guidance.solve_happened_this_step ? 1 : 0);

    // q_des 倾角
    float qd[4] = {fc.guidance.q_des[0], fc.guidance.q_des[1],
                   fc.guidance.q_des[2], fc.guidance.q_des[3]};
    // tilt = angle between Xb and -Zn (up)
    // Xb_n = C_bn * [1,0,0] = [1-2(qy²+qz²), 2(qxqy+qwqz), 2(qxqz-qwqy)]
    float qx = qd[1], qy = qd[2], qz = qd[3], qw = qd[0];
    float Xb_n[3] = {
        1.0f - 2.0f * (qy * qy + qz * qz),
        2.0f * (qx * qy + qw * qz),
        2.0f * (qx * qz - qw * qy)
    };
    float tilt_v = std::abs(Xb_n[2]);
    if (tilt_v > 1.0f) tilt_v = 1.0f;
    float tilt_rad = std::acos(tilt_v);  // angle from vertical
    float tilt_deg = tilt_rad * 180.0f / 3.14159265358979f;
    printf("q_des tilt_deg = %.4f\n", tilt_deg);

    // === 4. Python vs C++ 对比 ===
    printf("\n--- 4. Python vs C++ 逐层对比 ---\n");

    // throttle
    float cpp_throttle = fc.guidance.throttle;
    float d_throttle = cpp_throttle - static_cast<float>(fs.ref_throttle);
    printf("[throttle]      Python=%.6f  C++=%.6f  diff=%.6f  %s\n",
           fs.ref_throttle, cpp_throttle, d_throttle,
           std::abs(d_throttle) < 0.01f ? "OK" : "*** DIVERGE ***");

    // q_des
    float d_qdes[4];
    float max_dq = 0;
    for (int i = 0; i < 4; ++i) {
        d_qdes[i] = fc.guidance.q_des[i] - static_cast<float>(fs.ref_q_des[i]);
        if (std::abs(d_qdes[i]) > max_dq) max_dq = std::abs(d_qdes[i]);
    }
    printf("[q_des]         Python=[%.8f,%.8f,%.8f,%.8f]\n",
           fs.ref_q_des[0], fs.ref_q_des[1], fs.ref_q_des[2], fs.ref_q_des[3]);
    printf("                C++   =[%.8f,%.8f,%.8f,%.8f]\n",
           fc.guidance.q_des[0], fc.guidance.q_des[1],
           fc.guidance.q_des[2], fc.guidance.q_des[3]);
    printf("                diff  =[%.2e,%.2e,%.2e,%.2e]  %s\n",
           d_qdes[0], d_qdes[1], d_qdes[2], d_qdes[3],
           max_dq < 1e-4f ? "OK" : "*** DIVERGE ***");

    // n_engines
    int cpp_ne = fc.guidance.n_engines_current;
    printf("[n_engines]     Python=%d  C++=%d  %s\n",
           fs.ref_n_engines, cpp_ne,
           cpp_ne == fs.ref_n_engines ? "OK" : "*** DIVERGE ***");

    // bang_mode
    printf("[bang_mode]     Python=%d  C++=%d  %s\n",
           fs.bang_mode, fc.guidance.bang_mode ? 1 : 0,
           (fc.guidance.bang_mode ? 1 : 0) == fs.bang_mode ? "OK" : "*** DIVERGE ***");

    // TVC gimbal
    float cpp_tvc[2] = {fc.latest_control.tvc_gimbal[0],
                        fc.latest_control.tvc_gimbal[1]};
    float d_tvc[2];
    for (int i = 0; i < 2; ++i) {
        d_tvc[i] = cpp_tvc[i] - static_cast<float>(fs.ref_tvc_gimbal[i]);
    }
    printf("[tvc_gimbal]    Python=[%.8f,%.8f] rad = [%.4f,%.4f] deg\n",
           fs.ref_tvc_gimbal[0], fs.ref_tvc_gimbal[1],
           fs.ref_tvc_gimbal[0] * 57.2958, fs.ref_tvc_gimbal[1] * 57.2958);
    printf("                C++   =[%.8f,%.8f] rad = [%.4f,%.4f] deg\n",
           cpp_tvc[0], cpp_tvc[1],
           cpp_tvc[0] * 57.2958, cpp_tvc[1] * 57.2958);
    printf("                diff  =[%.2e,%.2e]  %s\n",
           d_tvc[0], d_tvc[1],
           std::abs(d_tvc[0]) < 1e-3f && std::abs(d_tvc[1]) < 1e-3f ? "OK" : "*** DIVERGE ***");

    // total_thrust
    float cpp_thrust = fc.latest_control.total_thrust;
    float d_thrust = cpp_thrust - static_cast<float>(fs.ref_total_thrust);
    printf("[total_thrust]  Python=%.4f  C++=%.4f  diff=%.4f  %s\n",
           fs.ref_total_thrust, cpp_thrust, d_thrust,
           std::abs(d_thrust) < 1000.0f ? "OK" : "*** DIVERGE ***");

    // === 5. G-FOLD 求解对比 (如果有) ===
    if (fs.has_ref_gfold && fc.guidance.solve_happened_this_step) {
        printf("\n--- 5. G-FOLD 求解对比 ---\n");
        float u0[3] = {fc.guidance.gfold.T_traj[0][0],
                       fc.guidance.gfold.T_traj[0][1],
                       fc.guidance.gfold.T_traj[0][2]};
        float T_mag = std::sqrt(u0[0]*u0[0] + u0[1]*u0[1] + u0[2]*u0[2]);
        printf("[u0]            Python=[%.4f,%.4f,%.4f]  C++=[%.4f,%.4f,%.4f]\n",
               fs.ref_gfold_u0[0], fs.ref_gfold_u0[1], fs.ref_gfold_u0[2],
               u0[0], u0[1], u0[2]);
        printf("[T_mag]         Python=%.4f  C++=%.4f  diff=%.4f\n",
               fs.ref_gfold_T_mag, T_mag, T_mag - static_cast<float>(fs.ref_gfold_T_mag));
    } else if (fc.guidance.solve_happened_this_step) {
        printf("\n--- 5. G-FOLD 求解 (C++ 有, Python 无参考) ---\n");
        float u0[3] = {fc.guidance.gfold.T_traj[0][0],
                       fc.guidance.gfold.T_traj[0][1],
                       fc.guidance.gfold.T_traj[0][2]};
        float T_mag = std::sqrt(u0[0]*u0[0] + u0[1]*u0[1] + u0[2]*u0[2]);
        printf("[u0]            C++=[%.4f,%.4f,%.4f]  T_mag=%.4f\n",
               u0[0], u0[1], u0[2], T_mag);
    } else {
        printf("\n--- 5. G-FOLD 求解: 本步无求解 (solve_counter%%100!=0) ---\n");
    }

    // === 6. 关键诊断 ===
    printf("\n--- 6. 关键诊断 ---\n");
    printf("slerp alpha     = %.6f  (t-last_solve=%.4f, interp_period=1.0)\n",
           (static_cast<float>(fs.t) - fc.guidance.last_solve_time) / 1.0f,
           static_cast<float>(fs.t) - fc.guidance.last_solve_time);
    printf("engines[0].thrust= %.4f N (TVC权限基准)\n",
           fc.octaweb.engines[0].thrust);
    printf("throttle_cmd    = %.6f\n", cpp_throttle);

    // bang-bang 分析
    float vz = static_cast<float>(fs.ekf_v[2]);
    float h = static_cast<float>(fs.h);
    float target_vz_bb;
    if (fc.guidance.n_engines_current >= 3) {
        float tmp = h / 20.0f;
        target_vz_bb = (tmp > guidance_params::VZ_TERMINAL_3) ? tmp : guidance_params::VZ_TERMINAL_3;
    } else {
        float h_pos = (h > 0.0f) ? h : 0.0f;
        float tmp = 2.0f * std::sqrt(h_pos);
        if (h > 200.0f && tmp > 50.0f) tmp = 50.0f;
        target_vz_bb = (tmp > guidance_params::VZ_TERMINAL_1) ? tmp : guidance_params::VZ_TERMINAL_1;
    }
    printf("vz              = %.4f m/s\n", vz);
    printf("target_vz_bb    = %.4f m/s\n", target_vz_bb);
    printf("deadband_bb     = %.4f m/s\n", guidance_params::DEADBAND_BB);
    if (vz > target_vz_bb + guidance_params::DEADBAND_BB) {
        printf("bang-bang state = TOO_FAST (throttle=T_min+Kp*(vz-target))\n");
    } else if (vz < target_vz_bb - guidance_params::DEADBAND_BB) {
        printf("bang-bang state = TOO_SLOW (C++: throttle=0.20, Python: throttle=0.0)\n");
    } else {
        printf("bang-bang state = DEADBAND (throttle=bang_last_throttle=%.6f)\n",
               fc.guidance.bang_last_throttle);
    }
}

// ===========================================================================
// 主函数
// ===========================================================================
int main(int argc, char* argv[]) {
    const char* frozen_file = "../golden_data/frozen_t10.txt";
    if (argc > 1) {
        frozen_file = argv[1];
    }

    printf("=== Golden Data 对齐测试 (C++ 端) ===\n");
    printf("冻结状态文件: %s\n", frozen_file);

    // 1. 解析冻结状态
    FrozenState fs;
    if (!parse_frozen_file(frozen_file, fs)) {
        return 1;
    }

    printf("解析成功: t=%.4f phase=%s solve_counter=%d\n",
           fs.t, fs.phase, fs.solve_counter);

    // 2. 创建 FlightComputer 并设置内部状态
    FlightComputer fc;
    setup_flight_computer(fc, fs);

    // 3. 执行单步 process_control
    process_control(fc);

    // 4. 打印对比结果
    print_comparison(fc, fs);

    return 0;
}
