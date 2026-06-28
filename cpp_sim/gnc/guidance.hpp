// =============================================================================
// guidance.hpp - 制导算法 (G-FOLD SOCP + 死区保底 + 状态机)
// 猎鹰9号火箭回收算法 C++ 翻译项目
// 对应 Python src/guidance.py
//
// 飞行剖面:
//   Phase 1 DESCENT  (h>1500m):  速度PD跟踪, 目标30m/s交给G-FOLD段
//   Phase 2 G-FOLD   (1500m→5m): SOCP联合优化推力大小+方向, slerp插值q_des
//   Phase 3 DEADBAND (h<5m):     死点控制, 目标2m/s, RCS锁定垂直
//   Phase 4 LANDED   (h<0.5m):   着陆完成
//
// 核心架构:
//   - 姿态控制器每0.01s主循环都执行, 绝不包在重求解if块里
//   - G-FOLD每1s(100步)重求解(MPC), slerp插值q_des, 线性插值油门
//   - 1-3-1发动机切换 (vz>55→3发, vz<40→1发, 滞回min_dwell=2.0s)
//   - hover-slam: target_vz = 2*sqrt(h)
//
// SOCP 求解器:
//   嵌入式环境无法运行 CVXPY/CLARABEL, C++ 版本使用解析 fallback:
//   - 正常情况: 基于能量守恒的解析推力剖面 (位置约束 → 加速度 → 推力)
//   - 不可行时: 退化到 PD 制导 (返回 false, 由 SafetyMonitor 接管)
//   - 保留 SOCP 接口, 供 CVXPYgen 生成的纯 C 代码替换
//
// 约法三章:
//   1. 零动态内存 (无 new/malloc/vector, 全部定长数组)
//   2. float32 默认
//   3. 算法保真: 严格对应 Python src/guidance.py
// =============================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include "../core/types.hpp"
#include "safety.hpp"   // rocket_params, SolveStatus
#include "octaweb.hpp"   // LandingProfile

namespace falcon9 {

// ---------------------------------------------------------------------------
// 制导参数 (对应 Python guidance.py + rocket_params.py)
// ---------------------------------------------------------------------------
namespace guidance_params {
    // G-FOLD 物理参数
    constexpr float T_MIN_FRAC     = 0.40f;                   // 最小油门 40% (Merlin 1D 节流下限)
    constexpr float THETA_GLIDE    = 0.6981317007977318f;     // 40° [rad] 1发指向锥
    constexpr float THETA_GLIDE_3  = 0.8726646259971648f;     // 50° [rad] 3发指向锥
    constexpr float PI_F           = 3.14159265358979323846f;
    constexpr float SQRT2_HALF     = 0.7071067811865476f;

    // Q_VERT = [√2/2, 0, √2/2, 0] (垂直姿态, b->n, Hamilton)
    constexpr float Q_VERT_W       = SQRT2_HALF;
    constexpr float Q_VERT_X       = 0.0f;
    constexpr float Q_VERT_Y       = SQRT2_HALF;
    constexpr float Q_VERT_Z       = 0.0f;

    // G-FOLD 求解器参数
    constexpr int   GFOLD_N_MAX    = 50;        // 最大步数
    constexpr float GFOLD_DT_ALIGN = 1.0f;      // dt 对齐周期 [s] (与重求解周期匹配)

    // 阶段转换高度
    constexpr float H_GFOLD_ENTRY   = 1500.0f;  // G-FOLD 入口高度 [m]
    constexpr float H_DEADBAND_ENTRY = 5.0f;    // DEADBAND 入口高度 [m]
    constexpr float H_LANDED         = 0.5f;    // 着陆判定高度 [m]

    // 终端约束
    constexpr float H_TERMINAL_1   = 5.0f;      // 1发模式终端高度 [m]
    constexpr float VZ_TERMINAL_1  = 3.0f;      // 1发模式终端速度 [m/s]
    constexpr float VZ_TERMINAL_3  = 40.0f;     // 3发模式终端速度 [m/s]

    // DESCENT 段参数
    constexpr float V_TARGET_HIGH  = 80.0f;     // h>2000m 目标速度 [m/s]
    constexpr float V_TARGET_LOW   = 30.0f;     // h<1000m 目标速度 [m/s]
    constexpr float KP_DESCENT_V   = 0.5f;      // DESCENT 速度 PD 增益
    constexpr float TILT_DESCENT_MAX = 5.0f;    // DESCENT 段最大倾角 [°]

    // DEADBAND 段参数
    constexpr float TARGET_VZ_DB   = 2.0f;      // 死区目标速度 [m/s]
    constexpr float DEADBAND_DB    = 0.5f;      // 死区宽度 [m/s]
    constexpr float KP_DB          = 0.12f;     // 死区 PD 增益
    constexpr float TILT_DB_MAX    = 8.0f;      // 死区段最大倾角 [°]
    constexpr float KP_DB_POS      = 0.08f;     // 死区水平位置增益
    constexpr float KP_DB_VEL      = 0.5f;      // 死区水平速度增益

    // bang-bang 参数 (T_min=40% > 重力, 无法悬停, 必须 bang-bang)
    constexpr float DEADBAND_BB    = 0.5f;      // bang-bang 死区 [m/s]
    constexpr float KP_BB_1        = 0.12f;     // 1发 bang-bang 增益
    constexpr float KP_BB_3        = 0.05f;     // 3发 bang-bang 增益

    // MPC 参数
    constexpr int   SOLVE_PERIOD   = 100;       // 100步=1s 重求解 (100Hz)
    constexpr float SOLVE_DT_INTERP = 1.0f;     // slerp/线性插值周期 [s]

    // 安全倾角限制 [°] (THETA_GLIDE + 2° 余量)
    constexpr float TILT_LIMIT_1   = 42.0f;     // 1发模式 (40°+2°)
    constexpr float TILT_LIMIT_3   = 52.0f;     // 3发模式 (50°+2°)
}  // namespace guidance_params

// ---------------------------------------------------------------------------
// 制导阶段枚举
// ---------------------------------------------------------------------------
enum class GuidancePhase : uint8_t {
    DESCENT  = 0,  // 高空减速 (h>1500m)
    GFOLD    = 1,  // G-FOLD SOCP 制导 (1500m→5m)
    DEADBAND = 2,  // 死区保底 (h<5m)
    LANDED   = 3,  // 着陆完成
};

// ===========================================================================
// 辅助函数: Hamilton 四元数乘法
// out = q1 ⊗ q2, q=[w,x,y,z]
// ===========================================================================
inline void quat_multiply_hamilton(const float q1[4], const float q2[4],
                                    float out[4]) {
    float w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
    float w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
    out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    out[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    out[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
}

// ===========================================================================
// 辅助函数: 四元数归一化 (就地)
// ===========================================================================
inline void quat_normalize_inplace(float q[4]) {
    float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n > 1e-15f) {
        float inv = 1.0f / n;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    }
}

// ===========================================================================
// 辅助函数: 从 NED 推力方向构造期望姿态四元数
// 对应 Python guidance.py q_des_from_thrust_dir
//
// 推力沿 +Xb, 火箭垂直时 Xb 指向 -Z_n (上). 故参考方向 up=[0,0,-1].
// q_des 使 Xb 在 n 系对齐 thrust_dir_n.
//
// 修复22c: 在 body 系算旋转轴 (从 [1,0,0] 到 C(Q_VERT)^T @ d)
//   C_qvert = [[0,0,1],[0,1,0],[-1,0,0]]
//   C_qvert^T = [[0,0,-1],[0,1,0],[1,0,0]]
//   d_body = C_qvert^T @ d = [-d[2], d[1], d[0]]
// ===========================================================================
inline void q_des_from_thrust_dir(const float thrust_dir_n[3],
                                   float q_des[4]) {
    // 归一化推力方向
    float d[3];
    float dn = std::sqrt(thrust_dir_n[0] * thrust_dir_n[0] +
                         thrust_dir_n[1] * thrust_dir_n[1] +
                         thrust_dir_n[2] * thrust_dir_n[2]);
    if (dn < 1e-15f) {
        q_des[0] = guidance_params::Q_VERT_W;
        q_des[1] = guidance_params::Q_VERT_X;
        q_des[2] = guidance_params::Q_VERT_Y;
        q_des[3] = guidance_params::Q_VERT_Z;
        return;
    }
    d[0] = thrust_dir_n[0] / dn;
    d[1] = thrust_dir_n[1] / dn;
    d[2] = thrust_dir_n[2] / dn;

    // d_body = C_qvert^T @ d = [-d[2], d[1], d[0]]
    float d_body[3] = {-d[2], d[1], d[0]};

    // x_ref = [1, 0, 0]
    // cos_a = dot(x_ref, d_body) = d_body[0]
    float cos_a = d_body[0];
    if (cos_a > 1.0f)  cos_a = 1.0f;
    if (cos_a < -1.0f) cos_a = -1.0f;

    // axis = cross([1,0,0], d_body) = [0, -d_body[2], d_body[1]]
    float axis[3] = {0.0f, -d_body[2], d_body[1]};
    float axis_n = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);

    if (cos_a > 1.0f - 1e-6f) {
        // 已对齐, 返回 Q_VERT
        // 注意: epsilon 用 1e-6 而非 1e-9, 因为 float32 的 ULP≈1.19e-7,
        // 1.0f-1e-9f 在 float32 中等于 1.0f, 导致检查失效.
        // (软约束修复后水平推力=0, thrust_dir=[0,0,-1], cos_a=1.0 暴露此 bug)
        q_des[0] = guidance_params::Q_VERT_W;
        q_des[1] = guidance_params::Q_VERT_X;
        q_des[2] = guidance_params::Q_VERT_Y;
        q_des[3] = guidance_params::Q_VERT_Z;
        return;
    }
    if (cos_a < -1.0f + 1e-6f) {
        // 180° 反向, 绕 body X 轴
        q_des[0] = 0.0f;
        q_des[1] = 1.0f;
        q_des[2] = 0.0f;
        q_des[3] = 0.0f;
        quat_normalize_inplace(q_des);
        return;
    }

    // 归一化轴
    float inv_an = 1.0f / axis_n;
    axis[0] *= inv_an;
    axis[1] *= inv_an;
    axis[2] *= inv_an;

    float angle = std::acos(cos_a);
    float half  = angle * 0.5f;
    float c = std::cos(half);
    float s = std::sin(half);

    // q_dev = [cos(angle/2), sin(angle/2)*axis]
    float q_dev[4] = {c, axis[0] * s, axis[1] * s, axis[2] * s};

    // q_des = Q_VERT ⊗ q_dev
    float q_vert[4] = {guidance_params::Q_VERT_W, guidance_params::Q_VERT_X,
                       guidance_params::Q_VERT_Y, guidance_params::Q_VERT_Z};
    quat_multiply_hamilton(q_vert, q_dev, q_des);
    quat_normalize_inplace(q_des);
}

// ===========================================================================
// 辅助函数: 球面线性插值 (slerp)
// 对应 Python guidance.py quat_slerp
// q0, q1 为单位四元数, t ∈ [0,1]
// ===========================================================================
inline void quat_slerp(const float q0[4], const float q1_in[4],
                        float t, float out[4]) {
    float q1[4] = {q1_in[0], q1_in[1], q1_in[2], q1_in[3]};

    float dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    if (dot > 1.0f)  dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;

    // 双覆盖处理: dot < 0 时取反 q1 以保证最短路径
    if (dot < 0.0f) {
        q1[0] = -q1[0]; q1[1] = -q1[1];
        q1[2] = -q1[2]; q1[3] = -q1[3];
        dot = -dot;
    }

    if (dot > 0.9995f) {
        // 近似平行: 线性插值 + 归一化
        out[0] = q0[0] + t * (q1[0] - q0[0]);
        out[1] = q0[1] + t * (q1[1] - q0[1]);
        out[2] = q0[2] + t * (q1[2] - q0[2]);
        out[3] = q0[3] + t * (q1[3] - q0[3]);
        quat_normalize_inplace(out);
        return;
    }

    float theta_0     = std::acos(dot);
    float sin_theta_0 = std::sin(theta_0);
    float theta       = theta_0 * t;
    float s0          = std::sin(theta_0 - theta) / sin_theta_0;
    float s1          = std::sin(theta) / sin_theta_0;

    out[0] = s0 * q0[0] + s1 * q1[0];
    out[1] = s0 * q0[1] + s1 * q1[1];
    out[2] = s0 * q0[2] + s1 * q1[2];
    out[3] = s0 * q0[3] + s1 * q1[3];
    quat_normalize_inplace(out);
}

// ===========================================================================
// 辅助函数: 体X轴与向上方向(世界-Z)的夹角 [rad]
// 对应 Python quaternion_utils.tilt_angle_from_vertical
// 剔除滚转干扰 (修复2): 直接算体X轴在n系中的指向, 与 up=[0,0,-1] 的夹角
// ===========================================================================
inline float tilt_angle_from_vertical_q(const float q[4]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    // C_bn 第一列 = 体X轴在n系中的指向
    // C[2,0] = 2*(x*z - w*y) = 体X轴在世界Z方向的分量
    float x_body_world_z = 2.0f * (x * z - w * y);
    // up = [0, 0, -1] (NED 向上)
    // cos_tilt = dot(x_body_world, up) = -x_body_world_z
    float cos_tilt = -x_body_world_z;
    if (cos_tilt > 1.0f)  cos_tilt = 1.0f;
    if (cos_tilt < -1.0f) cos_tilt = -1.0f;
    return std::acos(cos_tilt);
}

// ===========================================================================
// G_FOLD_Planner - G-FOLD 着陆轨迹优化 (SOCP, 6D状态+固定质量)
//
// 对应 Python guidance.py G_FOLD_Planner 类
//
// 状态: x = [pos_n(3), vel_n(3)], NED
// 控制: u = T_n(3), NED 推力矢量
//
// C++ 版本: 解析 fallback (基于能量守恒/位置约束)
//   - 从位置约束推导所需加速度: a = 2*(pos_target - pos0 - vel0*tgo) / tgo²
//   - 推力: T = m*(a - g - F_aero/m)
//   - 幅值限幅 [T_min, T_max], 指向锥约束
//   - 零阶保持填充轨迹 (MPC 只执行首步)
// ===========================================================================
class G_FOLD_Planner {
public:
    static constexpr int N_MAX = guidance_params::GFOLD_N_MAX;

    int   N;                      // 最大步数参数
    float theta_max;              // 指向锥角度 [rad] (1发模式)
    float lambda_mag;             // 推力幅值惩罚 (保留接口, 解析版未使用)

    // 求解结果 (零阶保持轨迹)
    float T_traj[N_MAX][3];       // 推力轨迹 [N_solved][3] (NED, N)
    float sigma_traj[N_MAX];      // 推力幅值轨迹 [N_solved] (N)
    int   N_solved;               // 实际求解步数

    // 求解器状态跟踪 (供 SafetyMonitor 查询)
    bool          last_solve_success;
    SolveStatus   last_solve_status;
    int           solve_count;
    int           fail_count;
    float         last_cost;

    // -----------------------------------------------------------------------
    // 构造
    // -----------------------------------------------------------------------
    G_FOLD_Planner(int N_ = 50,
                   float theta_max_rad = guidance_params::THETA_GLIDE,
                   float lambda_mag_ = 0.0f)
        : N(N_ < 1 ? 1 : N_), theta_max(theta_max_rad), lambda_mag(lambda_mag_) {
        reset();
    }

    // -----------------------------------------------------------------------
    // 求解 G-FOLD (解析 fallback)
    //
    // 对应 Python G_FOLD_Planner.solve
    //
    // 参数:
    //   pos0[3]:       当前位置 (NED)
    //   vel0[3]:       当前速度 (NED)
    //   pos_target[3]: 终端目标位置 (NED)
    //   vel_target[3]: 终端目标速度 (NED)
    //   m_current:     当前质量 [kg]
    //   tgo:           剩余时间 [s]
    //   F_aero_n[3]:   气动力 (NED, N), 零阶保持
    //   n_engines:     发动机数 (1 或 3)
    //
    // 返回: true=成功, false=失败 (触发 PD fallback)
    // -----------------------------------------------------------------------
    bool solve(const float pos0[3], const float vel0[3],
               const float pos_target[3], const float vel_target[3],
               float m_current, float tgo,
               const float F_aero_n[3], int n_engines) {
        // 输入检查
        if (tgo < 0.1f || m_current < 1.0f) {
            last_solve_success = false;
            last_solve_status  = SolveStatus::OTHER_FAILURE;
            fail_count++;
            solve_count++;
            return false;
        }

        // dt 对齐重求解周期 1.0s, N 上限 50, 下限 1
        float dt = tgo / static_cast<float>(N);
        if (dt < guidance_params::GFOLD_DT_ALIGN) {
            dt = guidance_params::GFOLD_DT_ALIGN;
        }
        int N_actual = static_cast<int>(tgo / dt + 0.5f);
        if (N_actual < 1) N_actual = 1;
        if (N_actual > N) N_actual = N;
        dt = tgo / static_cast<float>(N_actual);  // 精确 dt 使 N*dt = tgo
        N_solved = N_actual;

        // 推力上下限 (按发动机数缩放)
        float h_current     = -pos0[2];
        float T_max_single  = rocket_params::thrust_at_alt(h_current);
        float T_max_phys    = static_cast<float>(n_engines) * T_max_single;
        float T_min_phys    = static_cast<float>(n_engines) *
                              guidance_params::T_MIN_FRAC * T_max_single;

        // 指向锥角度 (3发模式用更大角度)
        float theta_max_eff = (n_engines >= 3) ? guidance_params::THETA_GLIDE_3
                                                : theta_max;

        // === 解析推力剖面 (基于位置约束 + SOCP软约束对齐) ===
        // 匀加速运动: pos(tgo) = pos0 + vel0*tgo + 0.5*a*tgo²
        // => a = 2*(pos_target - pos0 - vel0*tgo) / tgo²
        //
        // SOCP软约束对齐 (与Python guidance.py L235/L237对齐):
        //   Python SOCP终端位置允许20m软约束, 速度允许10m/s软约束,
        //   且燃料最小化(sum(sigma))自然限制水平推力.
        //   解析解无燃料最小化, 用软约束近似:
        //   - 水平位置误差 < 20m: 不修正 (软约束内, 与SOCP行为一致)
        //   - 水平位置误差 > 20m: 只修正超出部分 (修正到20m球边)
        //   - 水平速度误差 > 10m/s: 附加速度修正 (防漂移出球)
        //   垂直方向: 全修正 (需制动, 误差远超20m, 软约束可忽略)
        //   根因: Golden Data对齐测试t=7.72s发现解析解水平推力4x于SOCP
        //         (C++=52978N vs Python=12839N), q_des tilt 2.70° vs 0.83°,
        //         形成正反馈: tilt→drift→error→tilt→发散.
        float e_pos[3];
        for (int i = 0; i < 3; ++i) {
            e_pos[i] = pos_target[i] - pos0[i] - vel0[i] * tgo;
        }

        // 水平位置软约束 (20m, 与Python SOCP对齐)
        constexpr float SOFT_POS_TOL = 20.0f;  // [m] guidance.py L235
        float e_pos_xy_mag = std::sqrt(e_pos[0]*e_pos[0] + e_pos[1]*e_pos[1]);
        float xy_pos_scale = 1.0f;
        if (e_pos_xy_mag > SOFT_POS_TOL) {
            xy_pos_scale = (e_pos_xy_mag - SOFT_POS_TOL) / e_pos_xy_mag;
        } else {
            xy_pos_scale = 0.0f;  // 已在软约束球内, 无需水平位置修正
        }

        // 水平速度软约束 (10m/s, 与Python SOCP对齐)
        constexpr float SOFT_VEL_TOL = 10.0f;  // [m/s] guidance.py L237
        float e_vel_xy[2] = {vel_target[0] - vel0[0], vel_target[1] - vel0[1]};
        float e_vel_xy_mag = std::sqrt(e_vel_xy[0]*e_vel_xy[0] +
                                       e_vel_xy[1]*e_vel_xy[1]);
        float xy_vel_scale = 0.0f;
        if (e_vel_xy_mag > SOFT_VEL_TOL) {
            xy_vel_scale = (e_vel_xy_mag - SOFT_VEL_TOL) / e_vel_xy_mag;
        }

        float a_req[3];
        float tgo2 = tgo * tgo;
        // 水平: 位置修正(到达20m球边) + 速度修正(减速到10m/s内)
        a_req[0] = 2.0f * e_pos[0] * xy_pos_scale / tgo2
                 + e_vel_xy[0] * xy_vel_scale / tgo;
        a_req[1] = 2.0f * e_pos[1] * xy_pos_scale / tgo2
                 + e_vel_xy[1] * xy_vel_scale / tgo;
        // 垂直: 全修正 (需制动, 误差远超软约束)
        a_req[2] = 2.0f * e_pos[2] / tgo2;

        // 推力: T = m*(a - g - F_aero/m) = m*a - m*g - F_aero
        // g_n = [0, 0, G0] (NED, +Z 向下)
        float T_req[3];
        T_req[0] = m_current * a_req[0] - F_aero_n[0];
        T_req[1] = m_current * a_req[1] - F_aero_n[1];
        T_req[2] = m_current * (a_req[2] - rocket_params::G0) - F_aero_n[2];

        // 推力幅值
        float T_mag = std::sqrt(T_req[0] * T_req[0] +
                                T_req[1] * T_req[1] +
                                T_req[2] * T_req[2]);

        // NaN 检查
        if (T_mag != T_mag) {  // NaN
            last_solve_success = false;
            last_solve_status  = SolveStatus::OTHER_FAILURE;
            fail_count++;
            solve_count++;
            return false;
        }

        // 幅值限幅 [T_min, T_max]
        // 注意: 不做 T_min 下限钳位! 与 Python SOCP 行为对齐:
        //   SOCP 返回 ||u||<T_min 时, bang_mode=True, 用 bang-bang 执行.
        //   若这里钳位到 T_min, gfold_control() 会看到 u_norm>=T_min → bang_mode=False,
        //   导致与 Python 分歧 (Golden Data 对齐测试 t=10.71s 发现此 bug).
        if (T_mag < 1.0f) {
            // 推力太小, 用 T_min 垂直向上 (数值稳定兜底)
            T_req[0] = 0.0f;
            T_req[1] = 0.0f;
            T_req[2] = -T_min_phys;
            T_mag = T_min_phys;
        } else if (T_mag > T_max_phys) {
            float scale = T_max_phys / T_mag;
            T_req[0] *= scale; T_req[1] *= scale; T_req[2] *= scale;
            T_mag = T_max_phys;
        }

        // 指向锥约束: |T_xy| <= tan(theta_max) * (-T_z)
        // 要求 T_z < 0 (推力向上, NED Z 向下)
        if (T_req[2] >= 0.0f) {
            // 推力不向上! 强制垂直向上用 T_min
            T_req[0] = 0.0f;
            T_req[1] = 0.0f;
            T_req[2] = -T_min_phys;
            T_mag = T_min_phys;
        } else {
            float T_xy_mag = std::sqrt(T_req[0] * T_req[0] +
                                       T_req[1] * T_req[1]);
            float tan_theta = std::tan(theta_max_eff);
            float max_xy = tan_theta * (-T_req[2]);
            if (T_xy_mag > max_xy && max_xy > 0.0f) {
                // 缩放水平分量以满足指向锥
                float scale = max_xy / T_xy_mag;
                T_req[0] *= scale;
                T_req[1] *= scale;
                // 重新计算幅值 (T_z 不变)
                T_mag = std::sqrt(T_req[0] * T_req[0] +
                                  T_req[1] * T_req[1] +
                                  T_req[2] * T_req[2]);
                // 不做 T_min 重钳位 (与 Python SOCP 对齐, bang_mode 判断需要原始 ||u||)
            }
        }

        // 填充轨迹 (零阶保持)
        for (int k = 0; k < N_actual; ++k) {
            T_traj[k][0] = T_req[0];
            T_traj[k][1] = T_req[1];
            T_traj[k][2] = T_req[2];
            sigma_traj[k] = T_mag;
        }

        // 代价 (近似: sum(sigma))
        last_cost = T_mag * static_cast<float>(N_actual);

        last_solve_success = true;
        last_solve_status  = SolveStatus::OPTIMAL;
        solve_count++;
        return true;
    }

    // -----------------------------------------------------------------------
    // 重置
    // -----------------------------------------------------------------------
    void reset() {
        N_solved = 0;
        last_solve_success = false;
        last_solve_status  = SolveStatus::OPTIMAL;
        solve_count = 0;
        fail_count  = 0;
        last_cost   = 0.0f;
        for (int k = 0; k < N_MAX; ++k) {
            T_traj[k][0] = 0.0f;
            T_traj[k][1] = 0.0f;
            T_traj[k][2] = 0.0f;
            sigma_traj[k] = 0.0f;
        }
    }
};

// ===========================================================================
// DeadbandController - 死区保底控制器
//
// 对应 Python guidance.py DeadbandController 类
//
// Step B 物理: T_min=40% → 338kN > 重力 275kN. 火箭无法悬停!
//   - throttle=T_min 时: 净上推力 63kN, 持续减速
//   - throttle=0 时: 重力加速下降
// 策略: bang-bang 控制 vz 在 [target-deadband, target+deadband] 内
//   - vz > target+deadband (太快): throttle=T_min, 减速
//   - vz < target-deadband (太慢): throttle=0, 重力加速
//   - 中间: 保持上一步 (滞回防抖)
//
// 水平修正: 小倾角 PD (最大 8°), 对抗风扰
// ===========================================================================
class DeadbandController {
public:
    float target_vz;       // 目标下降速度 [m/s]
    float deadband;        // 死区宽度 [m/s]
    float last_throttle;   // 上一步油门 (滞回用)

    // -----------------------------------------------------------------------
    // 构造
    // -----------------------------------------------------------------------
    DeadbandController(float target_vz_ = guidance_params::TARGET_VZ_DB,
                       float deadband_ = guidance_params::DEADBAND_DB)
        : target_vz(target_vz_), deadband(deadband_) {
        last_throttle = guidance_params::T_MIN_FRAC;
    }

    // -----------------------------------------------------------------------
    // 更新: bang-bang 油门 + 水平 PD 修正
    //
    // 对应 Python DeadbandController.update
    //
    // 参数:
    //   pos_n[3], vel_n[3]: 当前位置/速度 (NED)
    //   m: 当前质量 [kg] (未使用, 保留接口)
    //   dt: 时间步长 [s] (未使用, 保留接口)
    // 输出:
    //   throttle_out: 油门 [0,1]
    //   q_des_out[4]: 期望姿态四元数
    //   omega_des_out[3]: 期望角速度 (始终为 0)
    // -----------------------------------------------------------------------
    void update(const float pos_n[3], const float vel_n[3],
                float m, float dt,
                float& throttle_out, float q_des_out[4],
                float omega_des_out[3]) {
        (void)m;   // 未使用
        (void)dt;  // 未使用

        float vz = vel_n[2];  // NED +Z 向下, 下降 vz>0

        // bang-bang 油门控制
        if (vz > target_vz + deadband) {
            // 太快: 加油门减速
            float t = guidance_params::T_MIN_FRAC +
                      guidance_params::KP_DB * (vz - target_vz);
            if (t > 1.0f) t = 1.0f;
            if (t < guidance_params::T_MIN_FRAC) t = guidance_params::T_MIN_FRAC;
            last_throttle = t;
        } else if (vz < target_vz - deadband) {
            // 太慢: 关机自由落体
            last_throttle = 0.0f;
        }
        // else: 保持 last_throttle (滞回防抖)
        throttle_out = last_throttle;

        // 水平位置 PD 修正 (最大 8°)
        float px = pos_n[0], py = pos_n[1];
        float vx = vel_n[0], vy = vel_n[1];
        float a_h[2] = {
            -guidance_params::KP_DB_POS * px - guidance_params::KP_DB_VEL * vx,
            -guidance_params::KP_DB_POS * py - guidance_params::KP_DB_VEL * vy
        };
        float a_mag = std::sqrt(a_h[0] * a_h[0] + a_h[1] * a_h[1]);
        float max_a = rocket_params::G0 *
                      std::sin(guidance_params::TILT_DB_MAX *
                               guidance_params::PI_F / 180.0f);
        if (a_mag > max_a) {
            float scale = max_a / a_mag;
            a_h[0] *= scale;
            a_h[1] *= scale;
            a_mag = max_a;
        }

        if (a_mag > 0.01f) {
            float tilt  = std::asin(a_mag / rocket_params::G0);
            float dir_h[2] = {a_h[0] / a_mag, a_h[1] / a_mag};
            float cos_t = std::cos(tilt);
            float sin_t = std::sin(tilt);
            // thrust_dir = up * cos(tilt) + [dir_h, 0] * sin(tilt)
            // up_n = [0, 0, -1]
            float thrust_dir[3] = {
                dir_h[0] * sin_t,
                dir_h[1] * sin_t,
                -cos_t
            };
            q_des_from_thrust_dir(thrust_dir, q_des_out);
        } else {
            q_des_out[0] = guidance_params::Q_VERT_W;
            q_des_out[1] = guidance_params::Q_VERT_X;
            q_des_out[2] = guidance_params::Q_VERT_Y;
            q_des_out[3] = guidance_params::Q_VERT_Z;
        }

        omega_des_out[0] = 0.0f;
        omega_des_out[1] = 0.0f;
        omega_des_out[2] = 0.0f;
    }

    // -----------------------------------------------------------------------
    // 重置
    // -----------------------------------------------------------------------
    void reset() {
        last_throttle = guidance_params::T_MIN_FRAC;
    }
};

// ===========================================================================
// LandingGuidance - 着陆制导状态机
//
// 对应 Python guidance.py LandingGuidance 类
//
// 四阶段着陆: DESCENT → G-FOLD → DEADBAND → LANDED
// 核心架构:
//   - 姿态控制器每步都执行 (由调用方实现)
//   - G-FOLD 每 1s (100步) 重求解 (MPC), slerp 插值 q_des
//   - 油门线性插值 (与 q_des slerp 同步), 防止重规划时跳变
//   - 1-3-1 发动机切换 (LandingProfile 决策)
//   - bang-bang 模式 (T_min > 重力, 无法悬停)
// ===========================================================================
class LandingGuidance {
public:
    // === 输出 (update 后由调用方读取) ===
    float throttle;           // 油门 [0,1]
    float q_des[4];           // 期望姿态四元数 [w,x,y,z]
    float omega_des[3];       // 期望角速度 [rad/s]
    float tvc_gimbal[2];      // TVC 万向架指令 [pitch, yaw] [rad]
    GuidancePhase phase;      // 当前阶段
    int   n_engines_current;  // 当前发动机数 (1 或 3)
    bool  landed;             // 着陆完成标志

    // === Phase 0: 动态着陆点 (T4 甲板运动) ===
    bool  landing_point_dynamic;
    float landing_point_n[3];

    // === 子控制器 ===
    G_FOLD_Planner    gfold;
    DeadbandController deadband;
    LandingProfile    landing_profile;

    // === MPC 状态 ===
    int   solve_counter;            // 步数计数
    int   solve_period;             // 重求解周期 (100步=1s)
    float last_solve_time;          // 上次求解时间 [s]
    bool  last_solve_success;       // 上次求解成功
    bool  solve_happened_this_step; // 本步是否触发求解

    // === q_des 平滑状态 (slerp 插值) ===
    float q_des_prev[4];            // 上次求解时的 q_des
    float q_des_current[4];         // 当前 q_des
    float q_des_target[4];          // 目标 q_des

    // === throttle 平滑状态 (线性插值) ===
    float throttle_prev;            // 上次求解时的 throttle
    float throttle_current;         // 当前 throttle
    float throttle_target;          // 目标 throttle

    // === bang-bang 状态 ===
    bool  bang_mode;                // bang-bang 模式标志
    float bang_last_throttle;       // bang-bang 上一步油门

    // === G-FOLD 入口状态 ===
    float gfold_t0;                 // G-FOLD 入口时间 [s]
    float gfold_tgo_fixed;          // G-FOLD 入口时计算的 tgo [s]
    float F_aero_n_prev[3];         // 上一步气动力 (NED, 零阶保持)

    // -----------------------------------------------------------------------
    // 构造
    // -----------------------------------------------------------------------
    LandingGuidance(int gfold_N = guidance_params::GFOLD_N_MAX,
                    float dt = 0.01f)
        : gfold(gfold_N, guidance_params::THETA_GLIDE, 0.0f) {
        (void)dt;
        reset();
    }

    // -----------------------------------------------------------------------
    // 主更新: 返回 throttle, q_des, omega_des, tvc_gimbal, phase
    //
    // 对应 Python LandingGuidance.update
    //
    // 参数:
    //   pos[3]:     当前位置 (NED)
    //   vel[3]:     当前速度 (NED)
    //   q[4]:       当前姿态四元数 [w,x,y,z] (b->n, Hamilton)
    //   fuel_mass:  剩余燃料 [kg]
    //   t:          当前时间 [s]
    //   dt:         时间步长 [s]
    // -----------------------------------------------------------------------
    void update(const float pos[3], const float vel[3], const float q[4],
                float fuel_mass, float t, float dt) {
        (void)dt;
        float h  = -pos[2];       // 高度 = -Z (NED Z 向下)
        float vz = vel[2];        // 下降速度 (NED +Z 向下, 下降 vz>0)
        float m  = rocket_params::mass_total(fuel_mass);

        // 初始化 TVC 输出
        tvc_gimbal[0] = 0.0f;
        tvc_gimbal[1] = 0.0f;

        // === 阶段转换 ===
        // 修复11: G-FOLD 入口从 1000m 提高到 1500m, 给水平修正更多时间
        if (phase == GuidancePhase::DESCENT &&
            h <= guidance_params::H_GFOLD_ENTRY) {
            phase = GuidancePhase::GFOLD;
            last_solve_time = -10.0f;
            gfold_t0 = t;
            solve_counter = 0;
            // tgo 公式: 匀减速时间 tgo = 2d/(v0+vt), d=h-h_terminal, vt=3m/s
            float vz_entry = (vz > 0.5f) ? vz : 0.5f;
            float tgo_kin = 2.0f * (h - guidance_params::H_TERMINAL_1) /
                            (vz_entry + guidance_params::VZ_TERMINAL_1);
            gfold_tgo_fixed = (tgo_kin < 10.0f) ? 10.0f :
                              (tgo_kin > 55.0f) ? 55.0f : tgo_kin;
        }

        // DEADBAND 入口 (h<=5m, 缩短无法悬停的 DEADBAND 段)
        if (phase == GuidancePhase::GFOLD &&
            h <= guidance_params::H_DEADBAND_ENTRY) {
            phase = GuidancePhase::DEADBAND;
        }

        // LANDED
        if (phase == GuidancePhase::DEADBAND &&
            h <= guidance_params::H_LANDED) {
            phase = GuidancePhase::LANDED;
            landed = true;
        }

        // === 各阶段逻辑 ===
        if (phase == GuidancePhase::DESCENT) {
            descent_control(h, vz, m, t, pos, vel);
            n_engines_current = 1;
            // 更新 q_des 平滑状态
            for (int i = 0; i < 4; ++i) {
                q_des_prev[i]    = q_des[i];
                q_des_current[i] = q_des[i];
                q_des_target[i]  = q_des[i];
            }
            omega_des[0] = 0.0f;
            omega_des[1] = 0.0f;
            omega_des[2] = 0.0f;
        }
        else if (phase == GuidancePhase::GFOLD) {
            gfold_control(pos, vel, q, m, t);
        }
        else if (phase == GuidancePhase::DEADBAND) {
            deadband.update(pos, vel, m, dt, throttle, q_des, omega_des);
            for (int i = 0; i < 4; ++i) {
                q_des_current[i] = q_des[i];
            }
            n_engines_current = 1;
        }
        else {  // LANDED
            throttle = 0.0f;
            q_des[0] = guidance_params::Q_VERT_W;
            q_des[1] = guidance_params::Q_VERT_X;
            q_des[2] = guidance_params::Q_VERT_Y;
            q_des[3] = guidance_params::Q_VERT_Z;
            omega_des[0] = 0.0f;
            omega_des[1] = 0.0f;
            omega_des[2] = 0.0f;
            n_engines_current = 0;
        }
    }

    const char* phase_str() const {
        switch (phase) {
            case GuidancePhase::DESCENT:  return "DESCENT";
            case GuidancePhase::GFOLD:    return "GFOLD";
            case GuidancePhase::DEADBAND: return "DEADBAND";
            case GuidancePhase::LANDED:   return "LANDED";
            default:                      return "UNKNOWN";
        }
    }

    // -----------------------------------------------------------------------
    // Phase 0: 设置动态着陆点 (T4 甲板运动)
    // 对应 Python LandingGuidance.set_landing_point
    //
    // 仅 1发模式 (精确着陆段) 使用动态着陆点.
    // 3发模式 (暴力制动段) 目标点仍为 h-100m, 不受甲板运动影响.
    // -----------------------------------------------------------------------
    void set_landing_point(const float pos_n[3]) {
        landing_point_n[0] = pos_n[0];
        landing_point_n[1] = pos_n[1];
        landing_point_n[2] = pos_n[2];
        landing_point_dynamic = true;
    }

    // -----------------------------------------------------------------------
    // Phase 0: 复位为固定原点着陆点
    // -----------------------------------------------------------------------
    void reset_landing_point() {
        landing_point_n[0] = 0.0f;
        landing_point_n[1] = 0.0f;
        landing_point_n[2] = -5.0f;
        landing_point_dynamic = false;
    }

    // -----------------------------------------------------------------------
    // 设置气动力 (由飞行控制器调用, 零阶保持传入 SOCP)
    // -----------------------------------------------------------------------
    void set_aero_force(const float F_aero_n[3]) {
        F_aero_n_prev[0] = F_aero_n[0];
        F_aero_n_prev[1] = F_aero_n[1];
        F_aero_n_prev[2] = F_aero_n[2];
    }

    // -----------------------------------------------------------------------
    // 重置所有状态
    // -----------------------------------------------------------------------
    void reset() {
        // 输出
        throttle = 0.0f;
        q_des[0] = guidance_params::Q_VERT_W;
        q_des[1] = guidance_params::Q_VERT_X;
        q_des[2] = guidance_params::Q_VERT_Y;
        q_des[3] = guidance_params::Q_VERT_Z;
        omega_des[0] = 0.0f;
        omega_des[1] = 0.0f;
        omega_des[2] = 0.0f;
        tvc_gimbal[0] = 0.0f;
        tvc_gimbal[1] = 0.0f;
        phase = GuidancePhase::DESCENT;
        n_engines_current = 1;
        landed = false;

        // Phase 0: 动态着陆点
        landing_point_dynamic = false;
        landing_point_n[0] = 0.0f;
        landing_point_n[1] = 0.0f;
        landing_point_n[2] = -5.0f;

        // 子控制器
        gfold.reset();
        deadband.reset();
        landing_profile.reset();

        // MPC 状态
        solve_counter = 0;
        solve_period  = guidance_params::SOLVE_PERIOD;
        last_solve_time = -10.0f;
        last_solve_success = false;
        solve_happened_this_step = false;

        // q_des 平滑
        q_des_prev[0] = guidance_params::Q_VERT_W;
        q_des_prev[1] = guidance_params::Q_VERT_X;
        q_des_prev[2] = guidance_params::Q_VERT_Y;
        q_des_prev[3] = guidance_params::Q_VERT_Z;
        for (int i = 0; i < 4; ++i) {
            q_des_current[i] = q_des_prev[i];
            q_des_target[i]  = q_des_prev[i];
        }

        // throttle 平滑
        throttle_prev    = 0.5f;
        throttle_current = 0.5f;
        throttle_target  = 0.5f;

        // bang-bang
        bang_mode = false;
        bang_last_throttle = guidance_params::T_MIN_FRAC;

        // G-FOLD 入口
        gfold_t0 = 0.0f;
        gfold_tgo_fixed = 20.0f;
        F_aero_n_prev[0] = 0.0f;
        F_aero_n_prev[1] = 0.0f;
        F_aero_n_prev[2] = 0.0f;
    }

private:
    // -----------------------------------------------------------------------
    // DESCENT 段控制: 高空减速
    // 对应 Python LandingGuidance._descent_control
    //
    // 速度 PD 跟踪, 目标 30m/s 交给 G-FOLD 段.
    // 修复13: 增加水平速度阻尼, 防止 px 在 DESCENT 段增长.
    // 修复5: NED 中 a_actual = G0 - T/m, T = m*(G0 - a_needed)
    // -----------------------------------------------------------------------
    void descent_control(float h, float vz, float m, float t,
                         const float pos_n[3], const float vel_n[3]) {
        (void)t;
        (void)pos_n;

        // 目标速度: 线性插值, h=2000m→80m/s, h=1000m→30m/s
        float v_target;
        if (h > 2000.0f) {
            v_target = guidance_params::V_TARGET_HIGH;
        } else if (h > 1000.0f) {
            v_target = guidance_params::V_TARGET_LOW +
                       50.0f * (h - 1000.0f) / 1000.0f;
        } else {
            v_target = guidance_params::V_TARGET_LOW;
        }

        // PD: a_needed = -Kp * v_err (v_err>0 时 a<0 即向上减速)
        float v_err = vz - v_target;
        float a_needed = -guidance_params::KP_DESCENT_V * v_err;
        // T = m*(G0 - a_needed) (修复5: 符号修正)
        float T_needed = m * (rocket_params::G0 - a_needed);
        float T_sl = rocket_params::thrust_at_alt(h);
        throttle = T_needed / T_sl;
        if (throttle > 1.0f) throttle = 1.0f;
        if (throttle < guidance_params::T_MIN_FRAC) throttle = guidance_params::T_MIN_FRAC;

        // 水平速度阻尼 (最大 5°, 对抗水平速度)
        float vx = vel_n[0], vy = vel_n[1];
        float v_horiz_mag = std::sqrt(vx * vx + vy * vy);
        if (v_horiz_mag > 1.0f) {
            // 倾斜角与水平速度成正比, 最大 5°, 方向与速度相反
            float tilt_deg = guidance_params::TILT_DESCENT_MAX;
            float tmp = v_horiz_mag * 0.3f;
            if (tmp < tilt_deg) tilt_deg = tmp;
            float tilt_rad = tilt_deg * guidance_params::PI_F / 180.0f;
            float horiz_dir[2] = {-vx / v_horiz_mag, -vy / v_horiz_mag};
            // thrust_dir = up + tilt_rad * [horiz_dir, 0]
            // up_n = [0, 0, -1]
            float thrust_dir[3] = {
                tilt_rad * horiz_dir[0],
                tilt_rad * horiz_dir[1],
                -1.0f
            };
            // 归一化
            float dn = std::sqrt(thrust_dir[0] * thrust_dir[0] +
                                 thrust_dir[1] * thrust_dir[1] +
                                 thrust_dir[2] * thrust_dir[2]);
            if (dn > 1e-15f) {
                thrust_dir[0] /= dn;
                thrust_dir[1] /= dn;
                thrust_dir[2] /= dn;
            }
            q_des_from_thrust_dir(thrust_dir, q_des);
        } else {
            q_des[0] = guidance_params::Q_VERT_W;
            q_des[1] = guidance_params::Q_VERT_X;
            q_des[2] = guidance_params::Q_VERT_Y;
            q_des[3] = guidance_params::Q_VERT_Z;
        }
    }

    // -----------------------------------------------------------------------
    // G-FOLD 段控制: SOCP 联合优化 + slerp/线性插值
    // 对应 Python LandingGuidance._gfold_control
    //
    // 核心铁律7: G-FOLD 每 1s (100步) 重求解 (MPC), 取首控制量执行.
    // 1.1 Step A: 油门从 SOCP 解 ||T_vec[0]||/T_max 提取.
    // E5: 1-3-1 发动机切换. 3发用于 vz>40 暴力制动, 1发用于精确着陆.
    // Step B: bang-bang 执行 (T_min=40% > 重力, 无法悬停).
    // -----------------------------------------------------------------------
    void gfold_control(const float pos_n[3], const float vel_n[3],
                       const float q[4], float m, float t) {
        float h  = -pos_n[2];
        float vz = vel_n[2];
        float T_max_single = rocket_params::thrust_at_alt(h);

        // === E5: 1-3-1 发动机配置决策 ===
        float vz_for_tgo = (vz > 0.5f) ? vz : 0.5f;
        float tgo_est = 2.0f * (h - guidance_params::H_TERMINAL_1) /
                        (vz_for_tgo + guidance_params::VZ_TERMINAL_1);
        if (tgo_est < 1.0f) tgo_est = 1.0f;
        if (tgo_est > 55.0f) tgo_est = 55.0f;

        int n_engines = landing_profile.decide_engine_config(h, vz, tgo_est, t);
        n_engines_current = n_engines;
        float T_max = static_cast<float>(n_engines) * T_max_single;

        // E6.3: 本步是否触发求解标志 (重置)
        solve_happened_this_step = false;

        // === G-FOLD SOCP 求解 (每 100 步 = 1s) ===
        if (solve_counter % solve_period == 0) {
            solve_happened_this_step = true;

            // E5: 终端目标随发动机配置变化
            float pos_target[3];
            float vel_target[3];
            if (n_engines >= 3) {
                // 3发模式: 暴力制动段, 不要求精确着陆
                // h_target = h-100m (给1发段留空间), vz_target=40 (切换阈值)
                pos_target[0] = 0.0f;
                pos_target[1] = 0.0f;
                pos_target[2] = -(h - 100.0f);
                vel_target[0] = 0.0f;
                vel_target[1] = 0.0f;
                vel_target[2] = guidance_params::VZ_TERMINAL_3;
            } else {
                // 1发模式: 精确着陆, h=5m, vz=3m/s (hover-slam)
                // Phase 0: 动态着陆点 (T4 甲板运动)
                if (landing_point_dynamic) {
                    pos_target[0] = landing_point_n[0];
                    pos_target[1] = landing_point_n[1];
                    pos_target[2] = -5.0f;
                } else {
                    pos_target[0] = 0.0f;
                    pos_target[1] = 0.0f;
                    pos_target[2] = -5.0f;
                }
                vel_target[0] = 0.0f;
                vel_target[1] = 0.0f;
                vel_target[2] = guidance_params::VZ_TERMINAL_1;
            }

            // 剩余时间 (匀减速 tgo)
            float vz_now = (vel_n[2] > 0.5f) ? vel_n[2] : 0.5f;
            float vz_terminal = vel_target[2];
            float tgo_remaining = 2.0f * (h - (-pos_target[2])) /
                                  (vz_now + vz_terminal);
            // E5: 1发模式 tgo 上限 20s (强制更快制动), 3发模式 55s
            float tgo_max = (n_engines == 1) ? 20.0f : 55.0f;
            if (tgo_remaining < 1.0f) tgo_remaining = 1.0f;
            if (tgo_remaining > tgo_max) tgo_remaining = tgo_max;

            // 求解
            bool success = gfold.solve(pos_n, vel_n, pos_target, vel_target,
                                       m, tgo_remaining, F_aero_n_prev,
                                       n_engines);

            if (success) {
                // 取首控制量
                float u0[3] = {gfold.T_traj[0][0],
                               gfold.T_traj[0][1],
                               gfold.T_traj[0][2]};
                float u_norm = std::sqrt(u0[0] * u0[0] +
                                         u0[1] * u0[1] +
                                         u0[2] * u0[2]);
                float T_min_phys = guidance_params::T_MIN_FRAC * T_max;

                // Step B: bang-bang 执行判断
                //   ||u|| >= T_min: 连续油门, throttle = ||u||/T_max
                //   ||u|| < T_min: bang-bang (0 或 T_min), 占空比平均 = ||u||
                // E5: 1发模式强制 bang-bang (MPC 燃料最优解会卡在低推力段)
                float throttle_new = 0.0f;
                if (n_engines == 1) {
                    bang_mode = true;
                } else if (u_norm >= T_min_phys - 1.0f) {
                    // -1N 容差防数值抖动
                    bang_mode = false;
                    throttle_new = u_norm / T_max;
                    if (throttle_new > 1.0f) throttle_new = 1.0f;
                    if (throttle_new < guidance_params::T_MIN_FRAC)
                        throttle_new = guidance_params::T_MIN_FRAC;
                } else {
                    bang_mode = true;
                }

                // 推力方向 → q_des (与 Python guidance.py L670-675 对齐)
                // 解析法 T_req 方向 = m*(a_req - g) - F_aero, 已做指向锥约束.
                // u0 = T_traj[0] = T_req (零阶保持), 方向即推力方向.
                float q_des_new[4];
                if (u_norm > 1.0f) {
                    float thrust_dir[3] = {
                        u0[0] / u_norm,
                        u0[1] / u_norm,
                        u0[2] / u_norm
                    };
                    q_des_from_thrust_dir(thrust_dir, q_des_new);
                } else {
                    q_des_new[0] = guidance_params::Q_VERT_W;
                    q_des_new[1] = guidance_params::Q_VERT_X;
                    q_des_new[2] = guidance_params::Q_VERT_Y;
                    q_des_new[3] = guidance_params::Q_VERT_Z;
                }

                // 更新 slerp/线性插值状态
                for (int i = 0; i < 4; ++i) {
                    q_des_prev[i]   = q_des_current[i];
                    q_des_target[i] = q_des_new[i];
                }
                if (!bang_mode) {
                    throttle_prev   = throttle_current;
                    throttle_target = throttle_new;
                }
                last_solve_time    = t;
                last_solve_success = true;
            } else {
                // 求解失败: 保持上一周期目标, 插值继续完成过渡
                last_solve_success = false;
            }
        }

        // === slerp 插值 q_des + 油门计算 ===
        float solve_dt = t - last_solve_time;
        float alpha = solve_dt / guidance_params::SOLVE_DT_INTERP;
        if (alpha > 1.0f) alpha = 1.0f;
        if (alpha < 0.0f) alpha = 0.0f;

        quat_slerp(q_des_prev, q_des_target, alpha, q_des);
        for (int i = 0; i < 4; ++i) {
            q_des_current[i] = q_des[i];
        }

        if (bang_mode) {
            // Step B: 连续 PD 油门 (T_min=40% > 重力, 无法悬停)
            // E5: 1发模式用 hover-slam 剖面 target_vz = 2*sqrt(h)
            //     3发模式目标 vz=40 (切换阈值)
            // 修复: 高空(h>200m)时 target_vz 上限 50 m/s, 防止从3发切换到1发后
            //   target远大于实际vz导致长时间throttle=0 (丧失TVC控制权)
            float target_vz_bb, Kp_bb;
            if (n_engines_current >= 3) {
                float tmp = h / 20.0f;
                target_vz_bb = (tmp > guidance_params::VZ_TERMINAL_3)
                               ? tmp : guidance_params::VZ_TERMINAL_3;
                Kp_bb = guidance_params::KP_BB_3;
            } else {
                float h_pos = (h > 0.0f) ? h : 0.0f;
                float tmp = 2.0f * std::sqrt(h_pos);
                // 高空限制: target_vz 不超过 50 m/s (避免3→1切换后target>>vz)
                // 低空(h<200m): 恢复标准 hover-slam 剖面
                if (h > 200.0f && tmp > 50.0f) {
                    tmp = 50.0f;
                }
                target_vz_bb = (tmp > guidance_params::VZ_TERMINAL_1)
                               ? tmp : guidance_params::VZ_TERMINAL_1;
                Kp_bb = guidance_params::KP_BB_1;
            }
            float deadband_bb = guidance_params::DEADBAND_BB;

            if (vz > target_vz_bb + deadband_bb) {
                // 太快: 加油门减速
                float thr = guidance_params::T_MIN_FRAC +
                            Kp_bb * (vz - target_vz_bb);
                if (thr > 1.0f) thr = 1.0f;
                if (thr < guidance_params::T_MIN_FRAC)
                    thr = guidance_params::T_MIN_FRAC;
                throttle = thr;
                bang_last_throttle = thr;
            } else if (vz < target_vz_bb - deadband_bb) {
                // 太慢: 关机让重力加速 (与 Python guidance.py L722 对齐)
                // 发动机关机尾推 (shutdown_tau=0.8s) 期间仍有推力, TVC有控制权.
                // 旧版用 0.20 导致 3发模式净推力>重力, 火箭持续减速, 卡在"太慢"正反馈.
                throttle = 0.0f;
                bang_last_throttle = 0.0f;
            } else {
                // 中间: 保持上一步 (滞回防抖)
                throttle = bang_last_throttle;
            }
            throttle_current = throttle;
        } else {
            // 连续模式: 线性插值
            throttle = (1.0f - alpha) * throttle_prev +
                       alpha * throttle_target;
            throttle_current = throttle;
        }

        // 安全保护: 倾角超限 → 强制回正
        // E5: 3发模式允许更大倾角 (50°+2°余量=52°), 1发模式 42°
        float tilt_limit_deg = (n_engines_current >= 3)
                               ? guidance_params::TILT_LIMIT_3
                               : guidance_params::TILT_LIMIT_1;
        float tilt_actual = tilt_angle_from_vertical_q(q);
        float tilt_deg = tilt_actual * 180.0f / guidance_params::PI_F;
        if (tilt_deg > tilt_limit_deg) {
            q_des[0] = guidance_params::Q_VERT_W;
            q_des[1] = guidance_params::Q_VERT_X;
            q_des[2] = guidance_params::Q_VERT_Y;
            q_des[3] = guidance_params::Q_VERT_Z;
            for (int i = 0; i < 4; ++i) {
                q_des_current[i] = q_des[i];
            }
        }

        omega_des[0] = 0.0f;
        omega_des[1] = 0.0f;
        omega_des[2] = 0.0f;

        // TVC gimbal: 由姿态环计算 (attitude_control), 制导环返回 0
        tvc_gimbal[0] = 0.0f;
        tvc_gimbal[1] = 0.0f;

        solve_counter++;
    }
};

}  // namespace falcon9
