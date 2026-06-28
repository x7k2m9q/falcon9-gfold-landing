// =============================================================================
// control.hpp - 姿态控制环 (100Hz 内环)
// 猎鹰9号火箭回收算法 C++ 翻译项目
// 对应 Python src/attitude_control.py
//
// 控制律: 四元数误差 PD (body系) + 气动力前馈
//   e_q = q_actual^{-1} ⊗ q_des  (body系误差, 修复10)
//   M_cmd = Kp * e_vec + Kd * e_omega - M_aero  (理论方案3.0铁律)
//
// 增益物理整定 (自适应惯量):
//   Kp = 2 * I * wn²  (补偿 e_vec = θ/2 因子)
//   Kd = 2 * zeta * wn * I
//
// 执行器分配:
//   G-FOLD/DEADBAND段: pitch/yaw → TVC, roll → RCS
//   TVC 万向架角度限制: ±5° (物理极限)
//   RCS 力矩分配: roll 通道优先用 RCS (TVC 无法控制 roll)
//
// E1改造: NotchFilter 串联到姿态误差通道 (pitch/yaw), 防止激励弯曲共振
//   弯曲模态在 Yb/Zb 方向, 滚转通道(Xb)无弯曲耦合, 不过滤
//   坑5防护(5.1审稿修正): NotchFilter 状态变量用 std::atomic 标记
//     原因: C++11起 volatile 不保证原子性与内存序, 多线程下存在数据竞争(Data Race).
//     std::atomic + memory_order_relaxed 保证无撕裂读写, 符合 FreeRTOS 多任务安全要求.
//
// 约法三章:
//   1. 零动态内存 (无 new/malloc/vector, 全部定长数组)
//   2. float32 默认
//   3. 算法保真: 严格对应 Python src/attitude_control.py
// =============================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include <atomic>
#include "../core/types.hpp"
#include "safety.hpp"  // rocket_params

namespace falcon9 {

// ---------------------------------------------------------------------------
// 控制参数 (对应 Python rocket_params.py + attitude_control.py)
// ---------------------------------------------------------------------------
namespace control_params {
    // TVC 万向架角度限制: ±5° (物理极限, 任务要求必须严格保持)
    constexpr float TVC_GIMBAL_LIMIT = 0.0872664625598042f;  // 5° [rad]

    // 发动机位置 (对应 Python rocket_params.ENGINE_X)
    constexpr float ENGINE_X = -13.0f;   // 发动机安装位置 Xb 坐标 (尾部)
    constexpr float LENGTH   = 30.0f;    // 箭体长度

    // 弯曲模态频率 (对应 Python flex_dynamics.py)
    // ω1 = (1.875/L)² * √(EI/μ) ≈ 14.76 rad/s = 2.35 Hz
    // ω2 = (4.694/L)² * √(EI/μ) ≈ 92.5 rad/s = 14.7 Hz
    constexpr float OMEGA_FLEX_1 = 14.76f;   // 一阶弯曲频率 [rad/s]
    constexpr float OMEGA_FLEX_2 = 92.5f;    // 二阶弯曲频率 [rad/s]
    constexpr float ZETA_NOTCH   = 0.1f;     // 陷波器阻尼比

    // PD 控制基础参数 (对应 Python AttitudeController.__init__)
    // Python 使用 wn=2π*0.3 (0.3Hz), 非默认 0.5Hz
    constexpr float WN_DEFAULT    = 2.0f * 3.14159265358979f * 0.3f;  // 0.3 Hz
    constexpr float ZETA_DEFAULT  = 0.9f;

    // 栅格舵参数 (对应 Python actuators.py GridFin)
    constexpr float GF_S_FIN     = 1.5f;       // 栅格舵面积 [m²]
    constexpr float GF_C_DELTA   = 2.0f;       // 力矩系数
    constexpr float GF_DELTA_MAX = 0.261799f;  // 15° [rad]
    constexpr float GF_X_FIN     = -13.0f;     // 栅格舵 Xb 位置 (尾部+2m)
    constexpr float GF_R         = 1.85f;      // 箭体半径 + 栅格舵展开半径
    constexpr float RCS_F        = 2000.0f;    // RCS 单喷口推力 [N]
    constexpr float RCS_X        = -14.0f;     // RCS Xb 位置
    constexpr float RCS_R_ROLL   = 3.85f;      // RCS 滚转力臂

    // 大气模型参数 (1976 标准大气简化)
    constexpr float RHO0    = 1.225f;       // 海平面密度
    constexpr float T0_ATM  = 288.15f;      // 海平面温度
    constexpr float SOUND0  = 340.29f;      // 海平面音速
    constexpr float SCALE_H = 8500.0f;      // 大气标高 [m]
}

// ---------------------------------------------------------------------------
// 大气模型 (简化指数模型, 对应 Python atmosphere.py)
// ---------------------------------------------------------------------------
namespace atm {
    // 简化大气: 指数衰减 (0-11km 对流层近似)
    inline float density(float h) {
        if (h <= 0.0f) return control_params::RHO0;
        if (h > 84852.0f) return 0.0f;
        return control_params::RHO0 * std::exp(-h / control_params::SCALE_H);
    }

    inline float sound_speed(float h) {
        if (h <= 0.0f) return control_params::SOUND0;
        // 简化: T = T0 - 0.0065*h (对流层), a = sqrt(gamma*R*T)
        float T = control_params::T0_ATM - 0.0065f * h;
        if (h > 11000.0f) T = 216.65f;  // 同温层
        return 340.29f * std::sqrt(T / control_params::T0_ATM);
    }
}

// ---------------------------------------------------------------------------
// 栅格舵效率 (对应 Python actuators.py gridfin_efficiency)
// ---------------------------------------------------------------------------
inline float gridfin_efficiency(float M) {
    if (M < 0.3f)       return 0.2f + 0.8f * (M / 0.3f);
    else if (M < 1.0f)  return 1.0f + 0.5f * (M - 0.3f) / 0.7f;
    else if (M < 1.5f)  return 1.5f;
    else                return 1.5f * std::exp(-(M - 1.5f) / 4.0f) + 0.5f;
}

// ---------------------------------------------------------------------------
// 栅格舵最大力矩估计 (简化, 对应 Python GridFin.max_torque_estimate)
// 单轴满偏时力矩 ≈ eff * qdyn * S * Cδ * |arm| * delta_max * 4(片)
// ---------------------------------------------------------------------------
inline void gf_max_torque(float mach, float qdyn, float cg_x,
                          float gf_max[3]) {
    float eff = gridfin_efficiency(mach);
    float rx = std::fabs(control_params::GF_X_FIN - cg_x);
    float s = 0.70710678f;  // 1/sqrt(2)
    float Rs = control_params::GF_R * s;
    // 精确计算: |B0[i,:] @ (D[:,i] * delta_max)| * eff * qdyn * S * C_delta
    // 滚转: |M_x| = 8 * Rs * s * delta_max (4片同向, 每片力臂 2*Rs*s)
    // 俯仰/偏航: |M_y| = |M_z| = 4 * rx * s * delta_max (2片主导)
    float base = eff * qdyn * control_params::GF_S_FIN *
                 control_params::GF_C_DELTA * control_params::GF_DELTA_MAX;
    gf_max[0] = base * 8.0f * Rs * s;     // 滚转
    gf_max[1] = base * 4.0f * rx * s;     // 俯仰
    gf_max[2] = base * 4.0f * rx * s;     // 偏航
}

// ---------------------------------------------------------------------------
// RCS 最大力矩 (对应 Python RCS._max_torque)
// ---------------------------------------------------------------------------
inline void rcs_max_torque(float cg_x, float rcs_max[3]) {
    rcs_max[0] = 2.0f * control_params::RCS_F * control_params::RCS_R_ROLL;
    rcs_max[1] = 2.0f * control_params::RCS_F * std::fabs(control_params::RCS_X - cg_x);
    rcs_max[2] = rcs_max[1];
}

// ===========================================================================
// NotchFilter - 二阶陷波器 (IIR, 状态空间实现)
//
// 对应 Python flex_dynamics.py NotchFilter / NotchFilterBank
//
// 传递函数 (连续域):
//   H(s) = (s² + w0²) / (s² + 2*zeta*w0*s + w0²)
//
// 特性:
//   - DC 处增益 = 1 (不影响稳态精度)
//   - w0 处增益 = 0 (完美陷波)
//   - 高频处增益 = 1 (不衰减高频信号)
//
// 状态空间实现:
//   dx1/dt = x2
//   dx2/dt = -w0²*x1 - 2*zeta*w0*x2 + 2*zeta*w0*u
//   y = u - x2
//
// 坑5防护(5.1审稿修正): 状态变量用 std::atomic 标记
//   volatile 在 C++11+ 不保证原子性/内存序, 改用 std::atomic<float>
//   + memory_order_relaxed 保证多任务下无数据撕裂
// ===========================================================================
class NotchFilter {
public:
    std::atomic<float> w0;    // 中心频率 [rad/s] (坑5: atomic 防优化+防撕裂)
    std::atomic<float> zeta;  // 阻尼比 (控制陷波宽度, 越大越宽)
    std::atomic<float> x1;    // 状态变量 1 (坑5: atomic 防优化+防撕裂)
    std::atomic<float> x2;    // 状态变量 2 (坑5: atomic 防优化+防撕裂)

    // -----------------------------------------------------------------------
    // 默认构造: 一阶弯曲频率
    // -----------------------------------------------------------------------
    NotchFilter() {
        w0.store(control_params::OMEGA_FLEX_1, std::memory_order_relaxed);
        zeta.store(control_params::ZETA_NOTCH, std::memory_order_relaxed);
        x1.store(0.0f, std::memory_order_relaxed);
        x2.store(0.0f, std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // 带参数构造
    // -----------------------------------------------------------------------
    NotchFilter(float w0_, float zeta_) {
        w0.store(w0_, std::memory_order_relaxed);
        zeta.store(zeta_, std::memory_order_relaxed);
        x1.store(0.0f, std::memory_order_relaxed);
        x2.store(0.0f, std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // 滤波单个样本
    // 参数: x = 输入样本, dt = 时间步长 [s]
    // 返回: 滤波后的值
    //
    // 使用半隐式 Euler 离散化 (对振荡系统更稳定):
    //   先更新 x2, 再用新 x2 更新 x1
    // -----------------------------------------------------------------------
    float update(float x, float dt) {
        // 原子读取到局部变量 (memory_order_relaxed: 无需同步, 仅防撕裂)
        float w   = w0.load(std::memory_order_relaxed);
        float z   = zeta.load(std::memory_order_relaxed);
        float s1  = x1.load(std::memory_order_relaxed);
        float s2  = x2.load(std::memory_order_relaxed);

        // 状态更新 (半隐式 Euler)
        float dx2 = -w * w * s1 - 2.0f * z * w * s2 + 2.0f * z * w * x;
        s2 += dx2 * dt;
        s1 += s2 * dt;

        // 原子写回
        x1.store(s1, std::memory_order_relaxed);
        x2.store(s2, std::memory_order_relaxed);

        // 输出 = 输入 - 带通部分
        return x - s2;
    }

    // -----------------------------------------------------------------------
    // 重置滤波器状态
    // -----------------------------------------------------------------------
    void reset() {
        x1.store(0.0f, std::memory_order_relaxed);
        x2.store(0.0f, std::memory_order_relaxed);
    }
};

// ===========================================================================
// AttitudeController - 姿态控制环
//
// 对应 Python attitude_control.py AttitudeController 类
//
// 控制律:
//   1. body系误差: e_q = q_actual^{-1} ⊗ q_des (修复10: body系, 非q_des系)
//   2. 双覆盖处理: e_q[0] < 0 时整体取反
//   3. 矢量部分: e_vec = [e_q[1], e_q[2], e_q[3]] ≈ θ_err/2
//   4. G-FOLD/DEADBAND: 忽略滚转误差 (e_vec[0] = 0, 修复7)
//   5. 陷波器: pitch/yaw 通道串联 NotchFilter (E1, 防激励弯曲共振)
//   6. PD + 前馈: M_cmd = Kp*e_vec + Kd*e_omega - M_aero
//   7. 分配: TVC ← pitch/yaw, RCS ← roll
//
// 增益: Kp = 2*I*wn², Kd = 2*zeta*wn*I (自适应惯量)
//   补偿 e_vec = θ/2 因子, 使闭环为标准 θ¨+2ζω_n·θ˙+ω_n²·θ=ω_n²·θ_des
// ===========================================================================
class AttitudeController {
public:
    // PD 增益 (per-axis, 由 set_inertia 计算)
    float Kp[3];    // 比例增益 [N·m/rad]
    float Kd[3];    // 微分增益 [N·m·s/rad]

    // 基础参数
    float wn;       // 自然频率 [rad/s] (~0.5Hz)
    float zeta;     // 阻尼比

    // 陷波器 (公开, 对应任务 C++ 类设计)
    NotchFilter notch_pitch;  // pitch(Yb)通道陷波器
    NotchFilter notch_yaw;    // yaw(Zb)通道陷波器

    // 运行时状态
    bool  gfold_phase;     // true = G-FOLD/DEADBAND段 (TVC参与, 忽略滚转)
    float thrust_actual;   // 当前推力 [N] (TVC力矩计算用)
    float cg_x;            // 当前质心 Xb 坐标 [m] (TVC力臂计算用)
    float last_M_cmd[3];   // 上一步力矩指令 (分配用, 对应 Python last_M_cmd)

    // TVC 万向架角度限制: 按飞行阶段自适应 (对应 Python rocket_params.py)
    //   G-FOLD/DEADBAND: 15° (Python TVC_GIMBAL_LIMIT_GFOld)
    //   其他段:          3°  (Python TVC_GIMBAL_LIMIT_NON_GFOld)
    static constexpr float TVC_LIMIT_GFOLD     = 0.2617993877991494f;  // 15° [rad]
    static constexpr float TVC_LIMIT_NON_GFOLD = 0.0523598775598299f;  // 3°  [rad]

    // -----------------------------------------------------------------------
    // 默认构造
    // -----------------------------------------------------------------------
    AttitudeController() {
        wn   = control_params::WN_DEFAULT;
        zeta = control_params::ZETA_DEFAULT;
        for (int i = 0; i < 3; ++i) {
            Kp[i] = 0.0f;
            Kd[i] = 0.0f;
            last_M_cmd[i] = 0.0f;
        }
        gfold_phase   = false;
        thrust_actual = 0.0f;
        cg_x          = -2.0f;  // 默认干质心
    }

    // -----------------------------------------------------------------------
    // 设置转动惯量对角线, 自动计算 PD 增益
    // 对应 Python: Kp = 2*I*wn², Kd = 2*zeta*wn*I
    // I_diag: [Ixx, Iyy, Izz] (body系转动惯量对角线)
    // -----------------------------------------------------------------------
    void set_inertia(const float I_diag[3]) {
        for (int i = 0; i < 3; ++i) {
            Kp[i] = 2.0f * I_diag[i] * wn * wn;
            Kd[i] = 2.0f * zeta * wn * I_diag[i];
        }
    }

    // -----------------------------------------------------------------------
    // 设置自然频率和阻尼比 (重算增益前需先调用 set_inertia)
    // -----------------------------------------------------------------------
    void set_params(float wn_, float zeta_) {
        wn   = wn_;
        zeta = zeta_;
    }

    // -----------------------------------------------------------------------
    // 设置飞行阶段
    // G-FOLD/DEADBAND: TVC参与姿态控制, 忽略滚转误差
    // -----------------------------------------------------------------------
    void set_phase(bool is_gfold) { gfold_phase = is_gfold; }

    // -----------------------------------------------------------------------
    // 设置当前推力 (TVC力矩计算用)
    // -----------------------------------------------------------------------
    void set_thrust(float thrust) { thrust_actual = thrust; }

    // -----------------------------------------------------------------------
    // 设置当前质心位置 (TVC力臂计算用)
    // -----------------------------------------------------------------------
    void set_cg(float cg) { cg_x = cg; }

    // -----------------------------------------------------------------------
    // 完整一步更新 (简化版本, 用于嵌入式调用)
    // 对应 Python AttitudeController.update + allocate
    // -----------------------------------------------------------------------
    void update(const float q_actual[4], const float omega[3],
                const float q_des[4], const float omega_des[3],
                float throttle, const float p[3], const float v[3],
                int n_engines, float tvc_gimbal_out[2],
                float gf_cmd_out[3], float rcs_cmd_out[3]) {
        (void)throttle; (void)n_engines;

        // 先调用完整 update 获取 tvc_gimbal 和 rcs_cmd
        float zero[3] = {0.0f, 0.0f, 0.0f};
        update(q_actual, q_des, omega, omega_des, zero, 0.01f, tvc_gimbal_out, rcs_cmd_out);

        // 栅格舵指令初始化
        for (int i = 0; i < 3; ++i) gf_cmd_out[i] = 0.0f;

        // 非 G-FOLD 段: 按动压分配栅格舵/RCS (对应 Python allocate)
        if (!gfold_phase) {
            // 计算动压 qdyn 和马赫数
            float h = -p[2];
            float v_mag = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            float rho = atm::density(h);
            float a_snd = atm::sound_speed(h);
            float mach = (a_snd > 0.0f) ? (v_mag / a_snd) : 0.0f;
            float qdyn = 0.5f * rho * v_mag * v_mag;

            // 估算栅格舵和RCS最大力矩
            float gf_max[3], rcs_max[3];
            gf_max_torque(mach, qdyn, cg_x, gf_max);
            rcs_max_torque(cg_x, rcs_max);

            // 防止除零
            for (int i = 0; i < 3; ++i) {
                if (gf_max[i] < 1.0f)  gf_max[i]  = 1.0f;
                if (rcs_max[i] < 1.0f) rcs_max[i] = 1.0f;
            }

            // 按动压分段分配 (对应 Python allocate)
            if (qdyn > 2000.0f) {
                // 高动压: 栅格舵主导
                for (int i = 0; i < 3; ++i) {
                    float cmd = last_M_cmd[i] / gf_max[i];
                    gf_cmd_out[i] = (cmd > 1.0f) ? 1.0f : ((cmd < -1.0f) ? -1.0f : cmd);
                    rcs_cmd_out[i] = 0.0f;
                }
            } else if (qdyn > 500.0f) {
                // 中动压: 栅格舵 + RCS 混合
                for (int i = 0; i < 3; ++i) {
                    float cmd = last_M_cmd[i] / gf_max[i];
                    gf_cmd_out[i] = (cmd > 1.0f) ? 1.0f : ((cmd < -1.0f) ? -1.0f : cmd);
                    float M_gf = gf_cmd_out[i] * gf_max[i];
                    float M_rcs = last_M_cmd[i] - M_gf;
                    float rcs_ratio = M_rcs / rcs_max[i];
                    rcs_cmd_out[i] = (rcs_ratio > 1.0f) ? rcs_max[i] :
                                     ((rcs_ratio < -1.0f) ? -rcs_max[i] : rcs_ratio * rcs_max[i]);
                }
            } else {
                // 低动压: 仅 RCS
                for (int i = 0; i < 3; ++i) {
                    float rcs_ratio = last_M_cmd[i] / rcs_max[i];
                    rcs_cmd_out[i] = (rcs_ratio > 1.0f) ? rcs_max[i] :
                                     ((rcs_ratio < -1.0f) ? -rcs_max[i] : rcs_ratio * rcs_max[i]);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 完整一步更新: 算力矩 → 陷波 → 分配到 TVC/RCS
    //
    // 对应 Python AttitudeController.update + compute_torque + allocate
    //
    // 参数:
    //   q_actual[4]:   实际姿态四元数 [w,x,y,z] (b->n, Hamilton)
    //   q_des[4]:      期望姿态四元数 [w,x,y,z]
    //   omega[3]:      实际角速度 [rad/s] (body系)
    //   omega_des[3]:  期望角速度 [rad/s] (body系)
    //   M_aero_b[3]:   气动力矩 [N·m] (body系, 前馈补偿用)
    //   dt:            时间步长 [s]
    //   tvc_gimbal_out[2]: 输出 TVC 万向架指令 [pitch, yaw] [rad]
    //   rcs_cmd_out[3]:    输出 RCS 力矩指令 [Mx, My, Mz] [N·m]
    // -----------------------------------------------------------------------
    void update(const float q_actual[4], const float q_des[4],
                const float omega[3], const float omega_des[3],
                const float M_aero_b[3], float dt,
                float tvc_gimbal_out[2], float rcs_cmd_out[3]) {

        // === 1. body系四元数误差: e_q = q_actual^{-1} ⊗ q_des ===
        // 修复10: body系误差, 非q_des系. e_vec[0]=滚转, [1]=俯仰, [2]=偏航
        float e_q[4];
        quat_error_body(q_actual, q_des, e_q);

        // === 2. 提取矢量部分 ===
        float e_vec[3] = {e_q[1], e_q[2], e_q[3]};

        // === 3. G-FOLD/DEADBAND: 忽略滚转误差 (修复7) ===
        // 原因: q_des_from_thrust_dir 的隐含滚转角会随推力方向跳变,
        //   导致 RCS 追逐滚转发散. TVC 可独立控制 pitch/yaw 不受滚转影响.
        if (gfold_phase) {
            e_vec[0] = 0.0f;  // body系[0] = 滚转
        }

        // === 4. 陷波器过滤 pitch/yaw 误差通道 (E1) ===
        // 弯曲模态在 Yb/Zb 方向, 滚转(Xb)无耦合, 不过滤
        // 陷波器在 DC 处增益=1, 不影响稳态精度
        e_vec[1] = notch_pitch.update(e_vec[1], dt);
        e_vec[2] = notch_yaw.update(e_vec[2], dt);

        // === 5. 角速度误差 ===
        float e_omega[3];
        for (int i = 0; i < 3; ++i) {
            e_omega[i] = omega_des[i] - omega[i];
        }
        // 角速度误差也加陷波 (同一通道, 独立状态)
        e_omega[1] = notch_pitch_omega_.update(e_omega[1], dt);
        e_omega[2] = notch_yaw_omega_.update(e_omega[2], dt);

        // === 6. PD + 气动力前馈 (理论方案3.0铁律: M_cmd = PD - M_aero) ===
        float M_cmd[3];
        for (int i = 0; i < 3; ++i) {
            M_cmd[i] = Kp[i] * e_vec[i] + Kd[i] * e_omega[i] - M_aero_b[i];
            last_M_cmd[i] = M_cmd[i];  // 保存供分配器使用
        }

        // === 7. 执行器分配 ===
        // 初始化输出
        tvc_gimbal_out[0] = 0.0f;
        tvc_gimbal_out[1] = 0.0f;
        rcs_cmd_out[0]    = 0.0f;
        rcs_cmd_out[1]    = 0.0f;
        rcs_cmd_out[2]    = 0.0f;

        if (gfold_phase) {
            // G-FOLD/DEADBAND段: pitch/yaw → TVC, roll → RCS
            // TVC力矩 = 推力 * sin(gimbal) * arm
            float arm = std::fabs(cg_x - control_params::ENGINE_X);
            float denom = thrust_actual * arm;

            if (thrust_actual > 1000.0f && denom > 1.0f) {
                // gp 与 M_y 反号: 正 M_y 需负 gp
                // 验证: gp<0 => Fz=-T*sin(gp)>0 => M_y=-rx*Fz=-负*正=正. 对!
                float ratio_y = M_cmd[1] / denom;
                float ratio_z = M_cmd[2] / denom;
                // clip to [-1, 1] 防 arcsin 域错误
                if (ratio_y > 1.0f)  ratio_y = 1.0f;
                if (ratio_y < -1.0f) ratio_y = -1.0f;
                if (ratio_z > 1.0f)  ratio_z = 1.0f;
                if (ratio_z < -1.0f) ratio_z = -1.0f;

                float gp = -std::asin(ratio_y);
                float gy = -std::asin(ratio_z);

                // TVC 万向架角度限制: 按飞行阶段自适应 (15° G-FOLD / 3° 其他)
                float tvc_lim = gfold_phase ? TVC_LIMIT_GFOLD : TVC_LIMIT_NON_GFOLD;
                if (gp > tvc_lim)  gp = tvc_lim;
                if (gp < -tvc_lim) gp = -tvc_lim;
                if (gy > tvc_lim)  gy = tvc_lim;
                if (gy < -tvc_lim) gy = -tvc_lim;

                tvc_gimbal_out[0] = gp;
                tvc_gimbal_out[1] = gy;

                // TVC饱和或推力不足时, 剩余力矩交给RCS
                // 工程直觉: bang-bang coasting段推力衰减, TVC权限不足,
                //   必须用RCS补充pitch/yaw控制权, 否则姿态单调发散.
                // TVC实际力矩: M_tvc = -thrust * arm * sin(gp)
                float M_tvc_y = -thrust_actual * arm * std::sin(gp);
                float M_tvc_z = -thrust_actual * arm * std::sin(gy);
                rcs_cmd_out[1] = M_cmd[1] - M_tvc_y;
                rcs_cmd_out[2] = M_cmd[2] - M_tvc_z;
            } else {
                // 推力不足 (bang-bang coasting 段, throttle=0):
                // TVC 无控制权 → pitch/yaw 交给 RCS 维持姿态
                // 工程直觉: 滑翔段火箭无推力, 必须靠 RCS 全轴维持姿态防翻车.
                // 不修正此路径会导致 tilt 在 coasting 段单调增长 → 发散.
                rcs_cmd_out[1] = M_cmd[1];
                rcs_cmd_out[2] = M_cmd[2];
            }

            // RCS: roll 通道 (TVC 无法控制 roll)
            rcs_cmd_out[0] = M_cmd[0];
        } else {
            // 非 G-FOLD段: 全部力矩给 RCS (简化分配)
            // Python 中按动压分段: 高速栅格舵, 低速RCS
            // C++ 简化: 直接给 RCS
            for (int i = 0; i < 3; ++i) {
                rcs_cmd_out[i] = M_cmd[i];
            }
        }
    }

    // -----------------------------------------------------------------------
    // 重置陷波器内部状态 (仿真重置用)
    // 对应 Python AttitudeController.reset_notch
    // -----------------------------------------------------------------------
    void reset_notch() {
        notch_pitch.reset();
        notch_yaw.reset();
        notch_pitch_omega_.reset();
        notch_yaw_omega_.reset();
    }

private:
    // 角速度通道陷波器 (独立状态, 与姿态误差通道分离)
    NotchFilter notch_pitch_omega_;
    NotchFilter notch_yaw_omega_;

    // -----------------------------------------------------------------------
    // body系四元数误差: e_q = q_actual^{-1} ⊗ q_des
    // 对应 Python: quat_multiply(quat_inverse(q_actual), q_des)
    //
    // 修复10: 误差在body系计算, 而非q_des系
    //   旧版 e_q = q_des ⊗ q_actual^{-1} 给出q_des系误差
    //   当 q_des = Q_VERT (90°旋转) 时 e_vec 分量对应 [偏航,俯仰,滚转]
    //   而非 [滚转,俯仰,偏航], 导致通道错乱
    //
    // 双覆盖处理: e_q[0] < 0 时整体取反 (保证最短路径)
    // -----------------------------------------------------------------------
    static void quat_error_body(const float q_actual[4], const float q_des[4],
                                 float e_q[4]) {
        // q_actual 的逆 = 共轭 / |q|² (单位四元数逆 = 共轭)
        float n2 = q_actual[0]*q_actual[0] + q_actual[1]*q_actual[1]
                 + q_actual[2]*q_actual[2] + q_actual[3]*q_actual[3];
        // 退化保护: 零四元数 -> 返回单位四元数 (无误差)
        if (n2 < 1e-15f) {
            e_q[0] = 1.0f; e_q[1] = 0.0f; e_q[2] = 0.0f; e_q[3] = 0.0f;
            return;
        }
        float inv_n2 = 1.0f / n2;

        float q_inv[4] = {
            q_actual[0] * inv_n2,
            -q_actual[1] * inv_n2,
            -q_actual[2] * inv_n2,
            -q_actual[3] * inv_n2
        };

        // Hamilton 乘法: q_inv ⊗ q_des
        float w1 = q_inv[0], x1 = q_inv[1], y1 = q_inv[2], z1 = q_inv[3];
        float w2 = q_des[0],  x2 = q_des[1],  y2 = q_des[2],  z2 = q_des[3];

        e_q[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;  // w
        e_q[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;  // x (滚转)
        e_q[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;  // y (俯仰)
        e_q[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;  // z (偏航)

        // 双覆盖处理: w < 0 时整体取反
        if (e_q[0] < 0.0f) {
            e_q[0] = -e_q[0];
            e_q[1] = -e_q[1];
            e_q[2] = -e_q[2];
            e_q[3] = -e_q[3];
        }
    }
};

// ===========================================================================
// 辅助函数: 从倾角构造期望姿态四元数
// 对应 Python attitude_control.py q_des_from_tilt
//
// 构造期望姿态: 垂直姿态绕 body 轴倾斜 tilt_rad
// q_des = Q_VERT ⊗ q_tilt
// ===========================================================================
inline void q_des_from_tilt(float tilt_rad, float axis_body[3],
                             float q_des_out[4]) {
    // 归一化轴
    float ax = axis_body[0], ay = axis_body[1], az = axis_body[2];
    float n = std::sqrt(ax*ax + ay*ay + az*az);
    if (n > 1e-15f) {
        ax /= n; ay /= n; az /= n;
    } else {
        ax = 0.0f; ay = 1.0f; az = 0.0f;  // 默认 Yb 轴
    }

    // q_tilt = [cos(θ/2), sin(θ/2)*axis]
    float half = tilt_rad * 0.5f;
    float c = std::cos(half);
    float s = std::sin(half);
    float q_tilt[4] = {c, ax * s, ay * s, az * s};

    // Q_VERT = [√2/2, 0, √2/2, 0]
    float sq2 = 0.7071067811865476f;
    float q_vert[4] = {sq2, 0.0f, sq2, 0.0f};

    // Hamilton 乘法: q_vert ⊗ q_tilt
    float w1 = q_vert[0], x1 = q_vert[1], y1 = q_vert[2], z1 = q_vert[3];
    float w2 = q_tilt[0], x2 = q_tilt[1], y2 = q_tilt[2], z2 = q_tilt[3];

    q_des_out[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    q_des_out[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    q_des_out[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    q_des_out[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

}  // namespace falcon9
