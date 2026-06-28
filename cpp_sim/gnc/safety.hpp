// =============================================================================
// safety.hpp - 安全监控器 + 异常处理框架 (工程硬化)
// 猎鹰9号火箭回收算法 C++ 翻译项目
// 对应 Python src/safety_monitor.py
//
// 理论方案2.0 E6节:
//   6.2 SafetyMonitor: G-FOLD 失效时的兜底决策
//     - 可达集检查 (能量法: vz² ≤ 2*a_brake*h)
//     - SOCP 不可行检测 (跟踪 last_solve_status)
//     - 燃料告警 (FUEL_RESERVE 阈值)
//     - 姿态超限 (tilt > 30° → RCS 全力回正)
//   6.3 FlightController: 异常处理框架 (分级兜底)
//
// 检查项 (优先级从高到低):
//   1. 姿态超限 (tilt > 30°): EMERGENCY_RECOVER — RCS 全力回正
//   2. 可达集违反 (vz² > 2*a_brake*h): ABORT_MAX_BRAKE — 全发满推力刹车
//   3. SOCP 不可行 (last_solve_status=infeasible): FALLBACK_PD — 退化 PD 控制
//   4. 燃料告警 (fuel < FUEL_RESERVE): MIN_FUEL_GLIDE — 最省油滑翔
//   5. 全部正常: NOMINAL
//
// 坐标系: NED (Z 向下为正), 下降 vz>0, 推力向上=u_z<0
//
// 约法三章:
//   1. 零动态内存 (无 new/malloc/vector, 全部定长数组)
//   2. float32 默认
//   3. 算法保真: 严格对应 Python src/safety_monitor.py
// =============================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include "../core/types.hpp"

namespace falcon9 {

// ---------------------------------------------------------------------------
// SOCP 求解器状态 (对应 Python safety_monitor.py 中 last_solve_status 字符串)
// ---------------------------------------------------------------------------
enum class SolveStatus : uint8_t {
    OPTIMAL              = 0,  // 求解成功
    INFEASIBLE           = 1,  // 不可行
    INFEASIBLE_INACCURATE= 2,  // 不可行 (不精确)
    OTHER_FAILURE        = 3,  // 其他失败
};

// ---------------------------------------------------------------------------
// 火箭物理参数 (对应 Python src/rocket_params.py, 仅保留 safety 所需项)
// ---------------------------------------------------------------------------
namespace rocket_params {
    constexpr float G0            = 9.80665f;     // 重力加速度 [m/s²]
    constexpr float DRY_MASS      = 22000.0f;     // 干重 [kg]
    constexpr float THRUST_SL     = 845000.0f;    // 海平面推力 [N]
    constexpr float THRUST_VAC    = 1200000.0f;   // 真空推力 [N]
    constexpr float THRUST_MIN_PCT= 0.40f;        // 最小油门 40%

    // 质心与转动惯量参数 (对应 Python rocket_params.py)
    constexpr float DRY_CG_X      = -2.0f;        // 干质心 Xb 坐标 [m]
    constexpr float FUEL_CG_X     = 1.5f;         // 燃料质心 Xb 坐标 [m]
    constexpr float IXX_DRY       = 5.0e4f;       // 干纵轴惯量 [kg·m²]
    constexpr float IYY_DRY       = 2.5e5f;       // 干横向惯量 [kg·m²]
    constexpr float IZZ_DRY       = 2.5e5f;       // 干横向惯量 [kg·m²]
    constexpr float ISP_SL        = 282.0f;       // 海平面比冲 [s]
    constexpr float ISP_VAC       = 311.0f;       // 真空比冲 [s]

    // 高度插值推力 (对应 rocket_params.thrust_at_alt)
    // h<0 按海平面, h>70km 按真空
    inline float thrust_at_alt(float h) {
        if (h <= 0.0f) return THRUST_SL;
        if (h >= 70000.0f) return THRUST_VAC;
        return THRUST_SL + (THRUST_VAC - THRUST_SL) * (h / 70000.0f);
    }

    // 总质量 (对应 rocket_params.mass_properties 返回的 m_total)
    inline float mass_total(float fuel_mass) {
        return DRY_MASS + fuel_mass;
    }

    // 比冲 (高度插值)
    inline float isp_at_alt(float h) {
        if (h <= 0.0f) return ISP_SL;
        if (h >= 70000.0f) return ISP_VAC;
        return ISP_SL + (ISP_VAC - ISP_SL) * (h / 70000.0f);
    }

    // 质量属性 (对应 Python rocket_params.mass_properties)
    // 返回 (m_total, cg_x, I_body[3]) via 输出参数
    inline void mass_properties(float fuel_mass,
                                float& m_total, float& cg_x, float I_body[3]) {
        m_total = DRY_MASS + fuel_mass;
        if (m_total < 1.0f) m_total = DRY_MASS;  // 退化保护
        cg_x = (DRY_MASS * DRY_CG_X + fuel_mass * FUEL_CG_X) / m_total;

        float d_dry = DRY_CG_X - cg_x;
        float Ixx = IXX_DRY;
        float Iyy = IYY_DRY + DRY_MASS * d_dry * d_dry;
        float Izz = IZZ_DRY + DRY_MASS * d_dry * d_dry;

        float d_fuel = FUEL_CG_X - cg_x;
        Ixx += 0.0f;  // 燃料纵轴惯量忽略
        Iyy += fuel_mass * d_fuel * d_fuel;
        Izz += fuel_mass * d_fuel * d_fuel;

        I_body[0] = Ixx;
        I_body[1] = Iyy;
        I_body[2] = Izz;
    }
}  // namespace rocket_params

// ===========================================================================
// SafetyMonitor - G-FOLD 失效时的兜底决策器
//
// 检查项 (优先级从高到低):
//   1. 姿态超限 (tilt > 30°): EMERGENCY_RECOVER
//   2. 可达集违反 (vz² > 2*a_brake*h): ABORT_MAX_BRAKE
//   3. SOCP 不可行: FALLBACK_PD
//   4. 燃料告警 (fuel < FUEL_RESERVE): MIN_FUEL_GLIDE
//   5. 全部正常: NOMINAL
// ===========================================================================
class SafetyMonitor {
public:
    // =======================================================================
    // 安全阈值 (对应 Python SafetyMonitor 类属性)
    // =======================================================================
    static constexpr float TILT_LIMIT_DEG          = 15.0f;   // 姿态超限阈值
    static constexpr float TILT_EMERGENCY_DEG      = 30.0f;   // 紧急姿态阈值 (RCS 全力)
    static constexpr float FUEL_RESERVE            = 500.0f;  // 燃料储备 [kg]
    static constexpr float FUEL_CRITICAL           = 100.0f;  // 临界燃料 [kg]
    static constexpr float VZ_MAX_DESCENT          = 120.0f;  // 最大下降速度 [m/s]
    static constexpr float H_MIN_BRAKE             = 50.0f;   // 最低制动高度 [m]

    // Phase 0: 推力一致性检查参数
    static constexpr float THRUST_FAULT_RATIO      = 0.10f;   // 10% 偏差阈值
    static constexpr int   THRUST_FAULT_STREAK_LIMIT = 5;     // 连续 5 步超限 → 确认故障
    static constexpr float THRUST_CHECK_MIN_EXPECTED = 500.0f;// 期望推力下限 [N]

    // 熔断阈值
    static constexpr int   FAIL_THRESHOLD_CIRCUIT  = 3;       // 连续 3 次失败熔断

    // =======================================================================
    // 状态成员 (公开, 便于外部读取)
    // =======================================================================
    SafetyStatus status;              // 当前安全状态
    SolveStatus  last_solve_status;   // 最近一次 SOCP 求解状态
    int          solve_fail_count;    // 累计失败次数
    int          solve_fail_streak;   // 连续失败次数 (用于熔断)
    bool         thrust_check_enabled;// 推力一致性检查使能 (默认关闭)
    int          thrust_fault_streak; // 推力故障连续计数
    float        last_thrust_ratio_error; // 最近一次推力偏差比
    char         last_violation[64];  // 最近一次违规描述 (调试用)

    // -----------------------------------------------------------------------
    // 默认构造
    // -----------------------------------------------------------------------
    SafetyMonitor() {
        status                  = SafetyStatus::NOMINAL;
        last_solve_status       = SolveStatus::OPTIMAL;
        solve_fail_count        = 0;
        solve_fail_streak       = 0;
        thrust_check_enabled    = false;  // 默认关闭, T2 测试时开启
        thrust_fault_streak     = 0;
        last_thrust_ratio_error = 0.0f;
        last_violation[0]       = '\0';
    }

    // -----------------------------------------------------------------------
    // 更新 SOCP 求解状态 (由 guidance 调用)
    // 对应 Python SafetyMonitor.update_solve_status()
    // -----------------------------------------------------------------------
    void update_solve_status(SolveStatus solve_status, bool success) {
        last_solve_status = solve_status;
        if (success) {
            solve_fail_streak = 0;
        } else {
            solve_fail_count++;
            solve_fail_streak++;
        }
    }

    // -----------------------------------------------------------------------
    // Phase 0: 推力一致性检查 (发动机硬故障检测)
    // 对应 Python SafetyMonitor.check_thrust_consistency()
    //
    // 坑2防护: 发动机故障"静默吞掉" — 必须主动检测
    //   判据: abs(T_actual - T_expected) > 0.1 * T_expected
    //   验收T2: 0.5s 内检测异常 → 连续 5 步 (0.05s@100Hz) 即触发
    //
    // 工程决策:
    //   1. 期望推力过小 (关机段/点火初期) 不检查, 避免误报
    //   2. 连续 5 步超限才确认, 防单步噪声抖动误触发
    //   3. 5 步@100Hz=0.05s, 远小于 T2 验收要求 0.5s
    //
    // 返回: true 表示检测到故障 (连续超限达阈值)
    // -----------------------------------------------------------------------
    bool check_thrust_consistency(float T_actual, float T_expected, float dt) {
        (void)dt;

        if (!thrust_check_enabled) {
            last_thrust_ratio_error = 0.0f;
            return false;
        }

        if (T_expected < THRUST_CHECK_MIN_EXPECTED) {
            thrust_fault_streak     = 0;
            last_thrust_ratio_error = 0.0f;
            return false;
        }

        last_thrust_ratio_error = std::fabs(T_actual - T_expected) / T_expected;

        if (last_thrust_ratio_error > THRUST_FAULT_RATIO) {
            thrust_fault_streak++;
        } else {
            thrust_fault_streak = 0;
        }

        return thrust_fault_streak >= THRUST_FAULT_STREAK_LIMIT;
    }

    void check_thrust_consistency(float T_actual, float T_expected, float dt,
                                  bool& is_fault, float& ratio, int& streak) {
        is_fault = check_thrust_consistency(T_actual, T_expected, dt);
        ratio = last_thrust_ratio_error;
        streak = thrust_fault_streak;
    }

    // -----------------------------------------------------------------------
    // Phase 0: 复位推力检查状态 (发动机重配后调用)
    // 对应 Python SafetyMonitor.reset_thrust_check()
    // -----------------------------------------------------------------------
    void reset_thrust_check() {
        thrust_fault_streak = 0;
    }

    // -----------------------------------------------------------------------
    // 执行所有安全检查, 返回安全状态
    // 对应 Python SafetyMonitor.check()
    //
    // 参数:
    //   state:     状态估计 (含 p, v, q)
    //   fuel:      剩余燃料 [kg]
    //   tgo:       估计剩余时间 [s] (保留接口, 当前未使用)
    //   n_engines: 当前发动机数 (影响制动能力)
    // -----------------------------------------------------------------------
    SafetyStatus evaluate(const StateEstimate& state, float fuel, float h,
                          float total_thrust, int n_engines) {
        (void)h; (void)total_thrust;
        return evaluate(state, fuel, 10.0f, n_engines);
    }

    SafetyStatus evaluate(const StateEstimate& state, float fuel,
                          float tgo = 10.0f, int n_engines = 1) {
        (void)tgo;  // 未使用, 保留接口兼容

        // === 0. 状态有效性检查 (工程硬化阶段一) ===
        // NaN/Inf 状态会绕过所有数值比较 (NaN > x 恒为 false),
        // 导致安全监控失效. 必须在最前面拦截.
        if (!is_state_valid(state)) {
            status = SafetyStatus::SAFE_MODE;
            std::snprintf(last_violation, sizeof(last_violation),
                          "state NaN/Inf detected");
            return status;
        }

        // 提取状态量
        float h  = -state.p[2];   // 高度 = -Z (NED Z 向下)
        float vz = state.v[2];    // 下降速度 (NED +Z 向下, 下降 vz>0)

        // === 1. 姿态超限检查 (最高优先级, 立即危及箭体结构) ===
        // tilt = 体X轴与向上方向(世界-Z)的夹角, 剔除滚转干扰
        float tilt_rad = tilt_angle_from_vertical(state.q);
        float tilt_deg = tilt_rad * (180.0f / PI_F);

        if (tilt_deg > TILT_EMERGENCY_DEG) {
            status = SafetyStatus::EMERGENCY_RECOVER;
            std::snprintf(last_violation, sizeof(last_violation),
                          "tilt=%.1f>%.1f (emergency)", tilt_deg, TILT_EMERGENCY_DEG);
            return status;
        }
        if (tilt_deg > TILT_LIMIT_DEG) {
            status = SafetyStatus::EMERGENCY_RECOVER;
            std::snprintf(last_violation, sizeof(last_violation),
                          "tilt=%.1f>%.1f", tilt_deg, TILT_LIMIT_DEG);
            return status;
        }

        // === 2. 可达集检查 (能量法) ===
        // 最大制动减速度: a_brake = T_max/m - g (向上为正, 减速下降)
        // 可达条件: vz² ≤ 2 * a_brake * h (动能 ≤ 势能差*制动能力)
        // 若 vz² > 2*a_brake*h → 当前推力无法在 h 内制动到 vz=0 → 必须 max_brake
        float m             = rocket_params::mass_total(fuel);
        float h_clamped     = (h > 0.0f) ? h : 0.0f;
        float T_max_single  = rocket_params::thrust_at_alt(h_clamped);
        float T_max         = static_cast<float>(n_engines) * T_max_single;
        float a_brake       = T_max / m - rocket_params::G0;  // 净向上加速度

        if (a_brake > 0.1f && h > H_MIN_BRAKE) {
            float vz_max_stoppable = std::sqrt(2.0f * a_brake * h);
            if (vz > vz_max_stoppable * 1.1f) {  // 10% 余量
                status = SafetyStatus::ABORT_MAX_BRAKE;
                std::snprintf(last_violation, sizeof(last_violation),
                              "vz=%.1f>%.1f (unreachable)", vz, vz_max_stoppable);
                return status;
            }
        }

        // 下降速度绝对上限 (超此值气动加热/结构载荷过大)
        if (vz > VZ_MAX_DESCENT) {
            status = SafetyStatus::ABORT_MAX_BRAKE;
            std::snprintf(last_violation, sizeof(last_violation),
                          "vz=%.1f>%.1f (limit)", vz, VZ_MAX_DESCENT);
            return status;
        }

        // === 3. SOCP 不可行检测 (连续失败熔断) ===
        if (solve_fail_streak >= FAIL_THRESHOLD_CIRCUIT) {
            status = SafetyStatus::FALLBACK_PD;
            std::snprintf(last_violation, sizeof(last_violation),
                          "SOCP fail streak=%d (circuit)", solve_fail_streak);
            return status;
        }

        if (last_solve_status == SolveStatus::INFEASIBLE ||
            last_solve_status == SolveStatus::INFEASIBLE_INACCURATE) {
            status = SafetyStatus::FALLBACK_PD;
            std::snprintf(last_violation, sizeof(last_violation),
                          "SOCP infeasible");
            return status;
        }

        // === 4. 燃料告警 ===
        if (fuel < FUEL_CRITICAL) {
            status = SafetyStatus::MIN_FUEL_GLIDE;
            std::snprintf(last_violation, sizeof(last_violation),
                          "fuel=%.0f<%.0f (critical)", fuel, FUEL_CRITICAL);
            return status;
        }
        if (fuel < FUEL_RESERVE) {
            status = SafetyStatus::MIN_FUEL_GLIDE;
            std::snprintf(last_violation, sizeof(last_violation),
                          "fuel=%.0f<%.0f", fuel, FUEL_RESERVE);
            return status;
        }

        // === 全部正常 ===
        status = SafetyStatus::NOMINAL;
        last_violation[0] = '\0';
        return status;
    }

    // -----------------------------------------------------------------------
    // 根据安全状态返回兜底油门指令
    // 对应 Python SafetyMonitor.get_fallback_throttle()
    //
    //   ABORT_MAX_BRAKE:   满推力 (1.0)
    //   EMERGENCY_RECOVER: 满推力 (1.0, 用推力矢量回正)
    //   MIN_FUEL_GLIDE:    最小推力 (THRUST_MIN_PCT, 省油滑翔)
    //   FALLBACK_PD / NOMINAL: 返回负值, 由调用方决定
    // -----------------------------------------------------------------------
    float get_fallback_throttle(SafetyStatus s) const {
        if (s == SafetyStatus::ABORT_MAX_BRAKE)   return 1.0f;
        if (s == SafetyStatus::EMERGENCY_RECOVER) return 1.0f;
        if (s == SafetyStatus::MIN_FUEL_GLIDE)    return rocket_params::THRUST_MIN_PCT;
        return -1.0f;  // FALLBACK_PD 或 NOMINAL, 由调用方决定
    }

private:
    // =======================================================================
    // 倾角计算 (与 Python quaternion_utils.tilt_angle_from_vertical 一致)
    //
    // 体X轴与向上方向(世界-Z)的夹角(弧度). 剔除滚转干扰(修复2).
    // 旧版 2*acos(|<q,Q_VERT>|) 会把纯滚转误算成倾斜, 导致 SAFE 误触发.
    // 新版: 直接算体X轴在n系中的指向, 与 up=[0,0,-1] 的夹角.
    //
    // 注意: 此处不使用 core/quaternion.hpp 的 Quaternion::tilt_angle_from_vertical(),
    //       因为该方法是简化版 (2*w*y), 仅在 x=0,z=0 时正确.
    //       Python 版是完整版, 剔除滚转干扰, 此处严格对应 Python.
    // =======================================================================
    static constexpr float PI_F = 3.14159265358979323846f;

    // -----------------------------------------------------------------------
    // 状态有效性检查 (工程硬化阶段一)
    // 检测 p, v, q 中是否存在 NaN/Inf.
    // NaN 会绕过所有数值比较, 导致安全监控失效, 必须主动检测.
    // -----------------------------------------------------------------------
    static bool is_state_valid(const StateEstimate& state) {
        // 位置
        for (int i = 0; i < 3; ++i) {
            if (std::isnan(state.p[i]) || std::isinf(state.p[i])) return false;
        }
        // 速度
        for (int i = 0; i < 3; ++i) {
            if (std::isnan(state.v[i]) || std::isinf(state.v[i])) return false;
        }
        // 四元数
        for (int i = 0; i < 4; ++i) {
            if (std::isnan(state.q[i]) || std::isinf(state.q[i])) return false;
        }
        // 四元数范数检查 (零四元数或范数异常)
        float qn = state.q[0]*state.q[0] + state.q[1]*state.q[1]
                 + state.q[2]*state.q[2] + state.q[3]*state.q[3];
        if (qn < 0.5f || qn > 2.0f) return false;
        return true;
    }

    static float tilt_angle_from_vertical(const float q[4]) {
        float w = q[0], x = q[1], y = q[2], z = q[3];

        // C_bn 旋转矩阵第一列 = 体X轴在 n 系中的指向
        // C[2,0] = 2(xz - wy)  (行主序下标 6)
        float x_body_world_z = 2.0f * (x * z - w * y);

        // up = [0, 0, -1] (NED 向上)
        float up_z = -1.0f;

        // cos_tilt = dot(x_body_world, up) = x_body_world_z * (-1)
        float cos_tilt = x_body_world_z * up_z;  // = -2(xz-wy) = 2(wy-xz)

        // clip to [-1, 1] 防止浮点误差导致 acos 域错误
        if (cos_tilt > 1.0f)  cos_tilt = 1.0f;
        if (cos_tilt < -1.0f) cos_tilt = -1.0f;

        return std::acos(cos_tilt);
    }
};

}  // namespace falcon9
