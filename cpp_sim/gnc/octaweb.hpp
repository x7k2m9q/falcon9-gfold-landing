// =============================================================================
// octaweb.hpp - Octaweb 发动机矩阵 (9台 Merlin 1D)
// 猎鹰9号火箭回收算法 C++ 翻译项目
// 对应 Python src/octaweb.py
//
// 理论方案2.0 E5节: 1-3-1 发动机配置
//   9台 Merlin 1D (8外围 + 1中心)
//   1发模式: 仅中心发动机 (有 TVC, 精确着陆)
//   3发模式: 中心 + 2台外围 (ACTIVE_3 = [0,1,5], 暴力制动)
//
// Phase 0: 发动机硬故障支持
//   failed_mask[9]: 故障发动机永久不可用
//   fail_engine(id): 注入故障, 推力立即归零
//   set_engine_config(): 跳过故障发动机
//   get_expected_thrust(): 期望推力 (一致性检查用)
//   last_total_thrust: 记录实际总推力
//
// 约法三章:
//   1. 零动态内存 (无 new/malloc/vector, 全部定长数组)
//   2. float32 默认
//   3. 算法保真: 严格对应 Python src/octaweb.py
// =============================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include "../core/types.hpp"
#include "safety.hpp"  // rocket_params (thrust_at_alt, G0, THRUST_MIN_PCT 等)

namespace falcon9 {

// ---------------------------------------------------------------------------
// Octaweb 物理参数 (对应 Python rocket_params.py, 仅保留 octaweb 所需项)
// ---------------------------------------------------------------------------
namespace octaweb_params {
    constexpr float DIAMETER       = 3.35f;              // 箭体直径 [m]
    constexpr float LENGTH         = 30.0f;              // 箭体长度 [m]
    constexpr float R_ENG          = DIAMETER * 0.5f * 0.7f;  // 发动机布局半径
    constexpr float SQRT2_HALF     = 0.7071067811865476f;     // √2/2

    // TVC 伺服参数 (对应 Python actuators.py TVC 类)
    constexpr float TVC_TAU        = 0.05f;              // TVC 一阶延迟 [s]
    constexpr float TVC_RATE_LIMIT = 0.5235987755982989f;     // 30°/s [rad/s]
    constexpr float TVC_LIMIT_GFOLD    = 0.2617993877991494f; // 15° [rad] G-FOLD段
    constexpr float TVC_LIMIT_NON_GFOLD= 0.0523598775598299f; // 3°  [rad] 非G-FOLD段

    // 发动机动态参数 (对应 Python EngineDynamics)
    constexpr float THRUST_TAU     = 0.3f;   // 油门响应时间常数 (涡轮泵惯性)
    constexpr float STARTUP_TIME   = 2.5f;   // 点火到满推力 [s]
    constexpr float SHUTDOWN_TAU   = 0.8f;   // 关机尾推时间常数 [s]
    constexpr float THRUST_NOISE_STD = 0.03f; // 推力抖动 ±3% (C++版不使用, 保留接口)

    // 发动机 X 位置 (尾部, 对应 Python rocket_params.ENGINE_X)
    constexpr float ENGINE_X       = -13.0f;
}  // namespace octaweb_params

// ===========================================================================
// EngineDynamics - 单台 Merlin 1D 发动机动态
//
// 对应 Python octaweb.py EngineDynamics 类
// 动态模型:
//   - 点火: S 曲线 (3t²-2t³) 上升, startup_time=2.5s
//   - 正常: 一阶滞后跟踪油门, thrust_tau=0.3s
//   - 关机: 指数衰减, shutdown_tau=0.8s
//   - thrust_ratio ∈ [0, 1], thrust = ratio * T_max
// ===========================================================================
struct EngineDynamics {
    int   id;                 // 发动机编号 [0-8]
    float thrust;             // 当前推力 [N]
    float thrust_ratio;       // 归一化推力 [0,1]
    float startup_progress;   // 点火进度 [0,1], 1=已完成
    bool  is_active;          // 是否点火活跃

    // 动态参数 (允许运行时调整, 默认与 Python 一致)
    float thrust_tau;         // 油门响应时间常数
    float startup_time;       // 点火时间
    float shutdown_tau;       // 关机尾推时间常数

    // -----------------------------------------------------------------------
    // 默认构造
    // -----------------------------------------------------------------------
    EngineDynamics() {
        id               = 0;
        thrust           = 0.0f;
        thrust_ratio     = 0.0f;
        startup_progress = 1.0f;  // 默认已完成 (已点火状态)
        is_active        = false;
        thrust_tau       = octaweb_params::THRUST_TAU;
        startup_time     = octaweb_params::STARTUP_TIME;
        shutdown_tau     = octaweb_params::SHUTDOWN_TAU;
    }

    // -----------------------------------------------------------------------
    // 更新发动机推力
    // 对应 Python EngineDynamics.update(throttle_cmd, h, dt, rng)
    // C++ 版: T_max 由调用方传入 (避免重复计算 thrust_at_alt)
    // throttle_cmd: [0,1] 直接推力百分比
    // T_max: 当前高度单台最大推力 [N]
    // dt: 时间步长 [s]
    // -----------------------------------------------------------------------
    void update(float throttle_cmd, float T_max, float dt) {
        float pct = throttle_cmd;
        if (pct > 1.0f) pct = 1.0f;
        if (pct < 0.0f) pct = 0.0f;

        if (pct > 0.01f) {
            // 有油门指令
            is_active = true;
            if (thrust_ratio < 0.01f && startup_progress < 1.0f) {
                // 点火 S 曲线: s(t) = 3t² - 2t³
                startup_progress += dt / startup_time;
                if (startup_progress > 1.0f) startup_progress = 1.0f;
                float t_norm = startup_progress;
                float s_curve = 3.0f * t_norm * t_norm - 2.0f * t_norm * t_norm * t_norm;
                float target = (pct > 0.4f) ? pct : 0.4f;  // 点火期至少 T_min(40%)
                thrust_ratio = s_curve * target;
            } else {
                // 正常运行: 一阶滞后跟踪油门
                float alpha = (thrust_tau > 1e-6f) ? (dt / thrust_tau) : 1.0f;
                if (alpha > 1.0f) alpha = 1.0f;
                thrust_ratio += alpha * (pct - thrust_ratio);
                startup_progress = 1.0f;
            }
        } else {
            // 油门指令=0: 关机尾推 (指数衰减)
            if (thrust_ratio > 0.01f) {
                float decay = (shutdown_tau > 1e-6f)
                    ? std::exp(-dt / shutdown_tau) : 0.0f;
                thrust_ratio *= decay;
            } else {
                thrust_ratio = 0.0f;
                is_active = false;
            }
            startup_progress = 0.0f;
        }

        // 实际推力 (C++ 版不加噪声, 嵌入式环境无 RNG)
        thrust = thrust_ratio * T_max;
    }

    // -----------------------------------------------------------------------
    // 重置发动机状态
    // -----------------------------------------------------------------------
    void reset() {
        thrust           = 0.0f;
        thrust_ratio     = 0.0f;
        startup_progress = 1.0f;
        is_active        = false;
        thrust_tau       = octaweb_params::THRUST_TAU;
        startup_time     = octaweb_params::STARTUP_TIME;
        shutdown_tau     = octaweb_params::SHUTDOWN_TAU;
    }
};

// ===========================================================================
// Octaweb - 9台 Merlin 1D 发动机矩阵
//
// 对应 Python octaweb.py Octaweb 类
// 布局: 8台外围(等间距45°) + 1台中心
//   中心(0号): 双向 TVC (pitch+yaw), 提供姿态控制
//   外围(1-8号): 固定推力方向, 仅轴向推力
//
// 1-3-1 策略:
//   3发模式: 中心(0) + 外围1,5 (对角, 推力对称) — ACTIVE_3 = {0, 1, 5}
//   1发模式: 仅中心(0) — ACTIVE_1 = {0}
// ===========================================================================
class Octaweb {
public:
    EngineDynamics engines[9];    // 9台发动机
    bool  active_mask[9];         // 当前活跃发动机掩码
    bool  failed_mask[9];         // Phase 0: 故障发动机 (永久不可用)
    int   n_active;               // 当前活跃发动机数
    float last_total_thrust;      // Phase 0: 上一步实际总推力 (一致性检查)

    // TVC 状态 (中心发动机, 对应 Python actuators.py TVC 类)
    float gimbal_pitch;           // 实际 pitch 偏角 [rad]
    float gimbal_yaw;             // 实际 yaw 偏角 [rad]
    float gimbal_pitch_prev;      // 上一步偏角 (速率限制用)
    float gimbal_yaw_prev;
    bool  gfold_phase;            // true=G-FOLD/DEADBAND段 (15°限幅), false=其他(3°)

    // -----------------------------------------------------------------------
    // 默认构造
    // -----------------------------------------------------------------------
    Octaweb() {
        reset();
    }

    // -----------------------------------------------------------------------
    // 设置活跃发动机数量 (1或3)
    // 对应 Python Octaweb.set_engine_config
    // Phase 0: 排除故障发动机
    // -----------------------------------------------------------------------
    void set_engine_config(int n_engines) {
        if (n_engines == 3) {
            for (int i = 0; i < 9; ++i) active_mask[i] = false;
            // ACTIVE_3 = {0, 1, 5}: 中心 + 对角两台外围
            static const int ACTIVE_3[3] = {0, 1, 5};
            for (int k = 0; k < 3; ++k) {
                int idx = ACTIVE_3[k];
                if (!failed_mask[idx]) {  // Phase 0: 跳过故障发动机
                    active_mask[idx] = true;
                }
            }
            n_active = 0;
            for (int i = 0; i < 9; ++i) {
                if (active_mask[i]) n_active++;
            }
        } else {
            // 1发模式: 仅中心
            for (int i = 0; i < 9; ++i) active_mask[i] = false;
            if (!failed_mask[0]) {  // 中心发动机未故障
                active_mask[0] = true;
            }
            n_active = 0;
            for (int i = 0; i < 9; ++i) {
                if (active_mask[i]) n_active++;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 0: 注入发动机硬故障 (永久不可用)
    // 对应 Python Octaweb.fail_engine
    // -----------------------------------------------------------------------
    void fail_engine(int engine_id) {
        if (engine_id < 0 || engine_id >= 9) return;
        failed_mask[engine_id] = true;
        // 如果该发动机当前活跃, 立即关闭
        if (active_mask[engine_id]) {
            active_mask[engine_id] = false;
            n_active = 0;
            for (int i = 0; i < 9; ++i) {
                if (active_mask[i]) n_active++;
            }
            // 强制推力归零
            engines[engine_id].thrust = 0.0f;
            engines[engine_id].thrust_ratio = 0.0f;
            engines[engine_id].is_active = false;
        }
    }

    // -----------------------------------------------------------------------
    // Phase 0: 返回给定油门下的期望总推力 (一致性检查用)
    // 对应 Python Octaweb.get_expected_thrust
    // -----------------------------------------------------------------------
    float get_expected_thrust(float throttle_cmd, float h) const {
        float T_max_single = rocket_params::thrust_at_alt(h);
        return static_cast<float>(n_active) * throttle_cmd * T_max_single;
    }

    // -----------------------------------------------------------------------
    // 返回当前配置的 (T_min, T_max)
    // 对应 Python Octaweb.get_thrust_bounds
    // -----------------------------------------------------------------------
    void get_thrust_bounds(float h, float& T_min, float& T_max) const {
        float T_max_single = rocket_params::thrust_at_alt(h);
        float T_min_single = rocket_params::THRUST_MIN_PCT * T_max_single;
        T_min = static_cast<float>(n_active) * T_min_single;
        T_max = static_cast<float>(n_active) * T_max_single;
    }

    // -----------------------------------------------------------------------
    // 设置当前飞行阶段 (影响 TVC 限幅)
    // -----------------------------------------------------------------------
    void set_phase(bool is_gfold) {
        gfold_phase = is_gfold;
    }

    // -----------------------------------------------------------------------
    // 更新所有发动机 + TVC 伺服
    // 对应 Python Octaweb.update(throttle_cmd, gimbal_pitch_cmd, gimbal_yaw_cmd,
    //                             phase, h, dt, rng)
    //
    // 参数:
    //   throttle:   G-FOLD 总油门指令 [0,1] (相对于单台 T_max)
    //   tvc_pitch:  中心发动机 TVC pitch 指令 [rad]
    //   tvc_yaw:    中心发动机 TVC yaw 指令 [rad]
    //   h:          当前高度 [m]
    //   dt:         时间步长 [s]
    // 返回: 总推力 [N]
    // -----------------------------------------------------------------------
    float update(float throttle, float tvc_pitch, float tvc_yaw,
                 const char* phase_name, float h, float dt) {
        (void)phase_name;
        return update(throttle, tvc_pitch, tvc_yaw, h, dt);
    }

    float update(float throttle, float tvc_pitch, float tvc_yaw,
                 float h, float dt) {
        float T_max_single = rocket_params::thrust_at_alt(h);

        // === TVC 伺服更新 (中心发动机, 对应 Python TVC.update) ===
        // 限幅
        float lim = gfold_phase ? octaweb_params::TVC_LIMIT_GFOLD
                                : octaweb_params::TVC_LIMIT_NON_GFOLD;
        float gp_cmd = tvc_pitch;
        if (gp_cmd > lim) gp_cmd = lim;
        if (gp_cmd < -lim) gp_cmd = -lim;
        float gy_cmd = tvc_yaw;
        if (gy_cmd > lim) gy_cmd = lim;
        if (gy_cmd < -lim) gy_cmd = -lim;

        // 一阶滞后
        float alpha_g = dt / octaweb_params::TVC_TAU;
        if (alpha_g > 1.0f) alpha_g = 1.0f;
        gimbal_pitch += (gp_cmd - gimbal_pitch) * alpha_g;
        gimbal_yaw   += (gy_cmd - gimbal_yaw)   * alpha_g;

        // 速率限制 30°/s
        float max_rate = octaweb_params::TVC_RATE_LIMIT * dt;
        float dp = gimbal_pitch - gimbal_pitch_prev;
        if (dp > max_rate) dp = max_rate;
        if (dp < -max_rate) dp = -max_rate;
        gimbal_pitch = gimbal_pitch_prev + dp;
        float dy = gimbal_yaw - gimbal_yaw_prev;
        if (dy > max_rate) dy = max_rate;
        if (dy < -max_rate) dy = -max_rate;
        gimbal_yaw = gimbal_yaw_prev + dy;

        gimbal_pitch_prev = gimbal_pitch;
        gimbal_yaw_prev   = gimbal_yaw;

        // === 中心发动机推力 (TVC 输出) ===
        // 中心发动机推力动态与外围相同, 但通过 TVC 偏转
        engines[0].update(throttle, T_max_single, dt);
        float total_thrust_center = engines[0].thrust;

        // === 外围发动机 (仅3发模式活跃) ===
        float outer_thrust = 0.0f;
        for (int i = 1; i < 9; ++i) {
            if (active_mask[i]) {
                // 外围发动机油门 = 中心油门 (均分总推力)
                engines[i].update(throttle, T_max_single, dt);
                outer_thrust += engines[i].thrust;
            } else {
                // 不活跃发动机: 关机尾推
                if (engines[i].thrust_ratio > 0.01f) {
                    engines[i].update(0.0f, T_max_single, dt);
                    outer_thrust += engines[i].thrust;
                } else {
                    engines[i].thrust = 0.0f;
                }
            }
        }

        float total = total_thrust_center + outer_thrust;
        last_total_thrust = total;  // Phase 0: 记录供一致性检查
        return total;
    }

    // -----------------------------------------------------------------------
    // 计算总推力(body系)和总力矩(body系)
    // 对应 Python Octaweb.compute_force_moment(gp, gy, cg_x)
    //
    // 中心发动机: 推力沿 TVC 偏转方向, 力矩 = 位置 × 力
    // 外围发动机: 推力沿 +Xb (固定), 力矩 = 位置 × 力 (仅 Yb/Zb 偏心)
    //
    // 参数:
    //   gp, gy: TVC 偏角 [rad]
    //   cg_x:   当前质心 Xb 坐标 [m]
    //   F_total_out[3]: 输出总推力 (body系) [N]
    //   M_total_out[3]: 输出总力矩 (body系) [N·m]
    // -----------------------------------------------------------------------
    void compute_force_moment(float gp, float gy, float cg_x,
                               float F_total_out[3], float M_total_out[3]) const {
        // 初始化
        for (int i = 0; i < 3; ++i) {
            F_total_out[i] = 0.0f;
            M_total_out[i] = 0.0f;
        }

        // 发动机 X 位置 (尾部)
        float x_tvc = -octaweb_params::LENGTH / 2.0f + 1.0f;
        float rx = x_tvc - cg_x;  // 发动机到质心的 X 距离

        // === 中心发动机 (TVC 偏转) ===
        if (engines[0].thrust > 100.0f) {
            float cp = std::cos(gp);
            float T0 = engines[0].thrust;
            // 推力方向 (body系): pitch 绕 Yb, yaw 绕 Zb
            float Fx = T0 * cp * std::cos(gy);
            float Fy = T0 * cp * std::sin(gy);
            float Fz = T0 * (-std::sin(gp));
            // 力矩 = r × F, r = [rx, 0, 0]
            // [0, -rx*Fz, rx*Fy]
            float My = -rx * Fz;
            float Mz = rx * Fy;
            F_total_out[0] += Fx;
            F_total_out[1] += Fy;
            F_total_out[2] += Fz;
            M_total_out[1] += My;
            M_total_out[2] += Mz;
        }

        // === 外围发动机 (固定方向, 沿 +Xb) ===
        for (int i = 1; i < 9; ++i) {
            if (engines[i].thrust > 100.0f) {
                float Fx = engines[i].thrust;
                float y_i, z_i;
                get_engine_pos_yz(i, y_i, z_i);
                // 力矩 = r × F, r = [rx, y_i, z_i], F = [Fx, 0, 0]
                // [0*0 - z_i*0, z_i*Fx - rx*0, rx*0 - y_i*Fx] = [0, z_i*Fx, -y_i*Fx]
                float My = z_i * Fx;
                float Mz = -y_i * Fx;
                F_total_out[0] += Fx;
                M_total_out[1] += My;
                M_total_out[2] += Mz;
            }
        }
    }

    // -----------------------------------------------------------------------
    // 获取发动机在 Yb-Zb 平面的位置 (相对几何中心)
    // 对应 Python Octaweb.ENGINE_POS_YZ
    // 中心在原点, 外围在半径 R_ENG 处等间距 45°
    // -----------------------------------------------------------------------
    static void get_engine_pos_yz(int idx, float& y, float& z) {
        const float R = octaweb_params::R_ENG;
        const float S = octaweb_params::SQRT2_HALF;
        switch (idx) {
            case 0: y = 0.0f;    z = 0.0f;    break;  // 中心
            case 1: y = R;       z = 0.0f;    break;  // +Y (右)
            case 2: y = R * S;   z = R * S;   break;  // +Y+Z
            case 3: y = 0.0f;    z = R;       break;  // +Z
            case 4: y = -R * S;  z = R * S;   break;  // -Y+Z
            case 5: y = -R;      z = 0.0f;    break;  // -Y (左)
            case 6: y = -R * S;  z = -R * S;  break;  // -Y-Z
            case 7: y = 0.0f;    z = -R;      break;  // -Z
            case 8: y = R * S;   z = -R * S;  break;  // +Y-Z
            default: y = 0.0f;   z = 0.0f;    break;
        }
    }

    // -----------------------------------------------------------------------
    // 重置所有状态
    // -----------------------------------------------------------------------
    void reset() {
        for (int i = 0; i < 9; ++i) {
            engines[i].id = i;
            engines[i].reset();
            active_mask[i] = false;
            failed_mask[i] = false;
        }
        active_mask[0] = true;  // 默认仅中心
        n_active = 1;
        last_total_thrust = 0.0f;
        gimbal_pitch = 0.0f;
        gimbal_yaw = 0.0f;
        gimbal_pitch_prev = 0.0f;
        gimbal_yaw_prev = 0.0f;
        gfold_phase = false;
    }
};

// ===========================================================================
// LandingProfile - 1-3-1 发动机点火策略
//
// 对应 Python octaweb.py LandingProfile 类
// E5改进: 基于 vz 的物理切换 (非 tgo), 增大滞回防抖
//   3发: vz > 55 (高速制动, T_min=1014kN 提供强减速 ~17m/s²)
//   1发: vz < 40 (精确着陆, T_min=338kN < 重力可 hover-slam)
//   滞回区: 40 < vz < 55 (保持当前配置, 防抖)
//   最小驻留时间: 2.0s (防快速切换)
//
// 物理依据:
//   3发 T_min=1014kN >> 重力363kN → 无法维持恒定 vz, 只能制动
//   必须在 vz 降到 40 时切换 1发, 否则 3发继续制动到 vz<0 (上升)
//   1发 T_min=338kN < 重力363kN → 可 hover-slam 精确着陆
// ===========================================================================
class LandingProfile {
public:
    int   current_config;       // 当前配置 (1 或 3)
    float vz_switch_high;       // 1→3 触发: vz > 55
    float vz_switch_low;        // 3→1 触发: vz < 40 (与 G-FOLD 3发终端 vz=40 对齐)
    float min_dwell;            // 最小驻留时间 [s]
    float last_switch_time;     // 上次切换时间 [s]

    // -----------------------------------------------------------------------
    // 默认构造
    // -----------------------------------------------------------------------
    LandingProfile() {
        reset();
    }

    // -----------------------------------------------------------------------
    // 决策发动机配置
    // 对应 Python LandingProfile.decide_engine_config(h, vz, tgo, t)
    // 返回: 应点火的发动机数量 (1 或 3)
    // -----------------------------------------------------------------------
    int decide_engine_config(float h, float vz, float tgo, float t) {
        (void)h;     // 未使用, 保留接口兼容
        (void)tgo;   // 未使用, 保留接口兼容

        // 驻留时间检查 (防快速切换)
        if (t - last_switch_time < min_dwell) {
            return current_config;
        }

        if (current_config == 3) {
            // 3发→1发: vz 降到 40 以下 (与 G-FOLD 3发终端 vz=40 对齐)
            if (vz < vz_switch_low) {
                current_config = 1;
                last_switch_time = t;
            }
        } else {
            // 1发→3发: vz 升到 55 以上
            if (vz > vz_switch_high) {
                current_config = 3;
                last_switch_time = t;
            }
        }

        return current_config;
    }

    // -----------------------------------------------------------------------
    // 重置
    // -----------------------------------------------------------------------
    void reset() {
        current_config  = 3;       // 初始 3发 (高速制动)
        vz_switch_high  = 55.0f;   // 1→3 触发
        vz_switch_low   = 40.0f;   // 3→1 触发
        min_dwell       = 2.0f;    // 最小驻留时间
        last_switch_time = -10.0f;
    }
};

}  // namespace falcon9
