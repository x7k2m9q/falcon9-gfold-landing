// =============================================================================
// scvx_solver.hpp - SCvx (Successive Convexification) 非线性制导求解器
// 猎鹰9号火箭回收算法 C++ 翻译项目 Phase 5B
//
// 对应 Python src/scvx_guidance.py (~870行)
//
// 核心创新 (相比G-FOLD):
//   1. 气动力一阶Taylor展开 (G-FOLD用零阶保持, 忽略速度-气动力耦合)
//   2. prox-linear迭代求解 (G-FOLD单次求解)
//   3. 信赖域保证线性化有效性
//   4. 虚拟控制保证SOCP可行性
//   5. 熔断机制: 8次迭代/1.2s超时 → G-FOLD降级
//   6. 多约束: 动压/热流/避障 (Phase 5A)
//
// 约法三章:
//   1. 零动态内存 (使用std::array, 禁止new/malloc/vector)
//   2. 强制使用double类型 (typedef double scalar_t, 避免float数值精度问题)
//   3. 预分配所有矩阵和向量 (编译期固定大小)
//
// OSQP接口:
//   通过宏 SCVX_USE_OSQP 控制. 如果定义了该宏且OSQP可用, 则使用OSQP求解SOCP;
//   否则降级到解析求解 (基于能量守恒的近似解, 与G-FOLD解析fallback类似).
//
// 坐标系: NED (Z向下为正), 下降vz>0, 推力向上=u_z<0
// =============================================================================
#ifndef FALCON9_SCVX_SOLVER_HPP
#define FALCON9_SCVX_SOLVER_HPP

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include "../core/fixed_matrix.hpp"
#include "../gnc/guidance.hpp"  // G_FOLD_Planner, SolveStatus, rocket_params

// OSQP接口条件编译
#ifdef SCVX_USE_OSQP
    // 当外部定义了SCVX_USE_OSQP时, 假设OSQP头文件已包含在编译路径中
    // #include "osqp.h"
    // #include "qdldl_interface.h"
#endif

namespace falcon9 {

// ===========================================================================
// 强制使用double (避免float数值精度问题)
// SCvx涉及Jacobian矩阵运算和信赖域判定, float32精度不足会导致收敛失败
// ===========================================================================
using scalar_t = double;

// ===========================================================================
// 编译期常量 (与Python scvx_guidance.py一致)
// ===========================================================================
constexpr std::size_t SCVX_N           = 30;  // 最大预测步数
constexpr std::size_t SCVX_STATE_DIM   = 6;   // 状态维度 (pos[3] + vel[3])
constexpr std::size_t SCVX_CONTROL_DIM = 3;   // 控制维度 (推力NED)

// ===========================================================================
// 定长类型别名 (基于FixedMatrix模板, 编译期固定大小)
// ===========================================================================
using StateVec   = FixedMatrix<scalar_t, SCVX_STATE_DIM, 1>;   // 6x1 状态向量
using ControlVec = FixedMatrix<scalar_t, SCVX_CONTROL_DIM, 1>; // 3x1 控制向量
using StateMat   = FixedMatrix<scalar_t, SCVX_STATE_DIM, SCVX_STATE_DIM>; // 6x6 状态矩阵
using InputMat   = FixedMatrix<scalar_t, SCVX_STATE_DIM, SCVX_CONTROL_DIM>; // 6x3 输入矩阵
using Vec3d      = FixedMatrix<scalar_t, 3, 1>;  // 3x1 向量
using Mat3d      = FixedMatrix<scalar_t, 3, 3>;  // 3x3 矩阵

// ===========================================================================
// 简化大气模型 (ISA对流层近似)
// 对应 Python atmosphere.py, 用于气动力线性化
// 高度范围: 0-11000m (着陆段主要范围)
// ===========================================================================
namespace scvx_atmos {
    constexpr scalar_t RHO0         = 1.225;      // 海平面密度 [kg/m³]
    constexpr scalar_t T0           = 288.15;     // 海平面温度 [K]
    constexpr scalar_t LAPSE_RATE   = 0.0065;     // 温度递减率 [K/m]
    constexpr scalar_t GAS_CONSTANT = 287.05;     // 空气气体常数 [J/(kg·K)]
    constexpr scalar_t G0           = 9.80665;    // 重力加速度 [m/s²]
    constexpr scalar_t RHO_THRESHOLD = 1e-6;      // 密度阈值 (低于此值视为真空)

    // 计算大气密度 ρ(h)
    // 使用ISA对流层模型: ρ = ρ0 * (T/T0)^(g/(R·L) - 1)
    inline scalar_t density(scalar_t h) {
        if (h <= 0.0) return RHO0;
        if (h >= 11000.0) {
            // 对流层顶以上: 指数衰减
            scalar_t T11   = T0 - LAPSE_RATE * 11000.0;
            scalar_t rho11 = RHO0 * std::pow(T11 / T0,
                                             G0 / (GAS_CONSTANT * LAPSE_RATE) - 1.0);
            scalar_t H_scale = GAS_CONSTANT * T11 / G0;
            return rho11 * std::exp(-(h - 11000.0) / H_scale);
        }
        // 对流层
        scalar_t T = T0 - LAPSE_RATE * h;
        return RHO0 * std::pow(T / T0, G0 / (GAS_CONSTANT * LAPSE_RATE) - 1.0);
    }

    // 计算密度对高度的梯度 ∂ρ/∂h (中心差分, 步长10m)
    // 用于多约束线性化 (动压/热流约束)
    inline scalar_t density_gradient(scalar_t h) {
        const scalar_t dh = 10.0;
        scalar_t rho_up = density(h + dh);
        scalar_t rho_dn = density(h - dh);
        return (rho_up - rho_dn) / (2.0 * dh);
    }
}  // namespace scvx_atmos

// ===========================================================================
// SCvxConfig - SCvx求解器配置参数
// 对应 Python SCvxPlanner 类属性
// ===========================================================================
struct SCvxConfig {
    // === 熔断参数 (Phase 5A放松以降低降级率) ===
    int      MAX_ITER;          // 最大prox-linear迭代次数 (8)
    scalar_t SOLVE_TIMEOUT;     // 单步SOCP求解超时 [s] (1.2)
    scalar_t TOTAL_TIMEOUT;     // 总求解超时 [s] (5.0)
    scalar_t CONV_TOL;          // 收敛阈值 (1e-2)

    // === 信赖域参数 ===
    // 限制状态偏离参考轨迹, 保证线性化有效性
    scalar_t TRUST_DELTA_POS;   // 位置信赖域 [m] (50)
    scalar_t TRUST_DELTA_VEL;   // 速度信赖域 [m/s] (10)

    // === 代价权重 ===
    scalar_t VIRTUAL_CONTROL_WEIGHT;  // 虚拟控制惩罚 (1e6)
    scalar_t TERMINAL_WEIGHT;         // 终端约束惩罚 (50000)
    scalar_t SMOOTH_WEIGHT;           // 推力方向平滑 (0.1)

    // === 多约束参数 (Phase 5A) ===
    scalar_t Q_MAX;               // 动压上限 [Pa] (80000)
    scalar_t Q_DOT_MAX;           // 热流上限 [W/m²] (1.5e6)
    scalar_t OBSTACLE_CENTER[3];  // 障碍物中心 [m] (50, 0, -50)
    scalar_t OBSTACLE_SIZE[3];    // 障碍物尺寸 [m] (30, 30, 100, 半尺寸=15,15,50)

    // === 气动模型参数 ===
    scalar_t CD_LINEARIZE;  // 线性化用阻力系数 (0.3)
    scalar_t REF_AREA;      // 参考面积 [m²] (π·(3.35/2)²)

    // === 指向锥角度 ===
    scalar_t theta_max;     // 推力指向锥角度 [rad] (40°=0.698)

    // 默认构造 (与Python一致)
    SCvxConfig()
        : MAX_ITER(8),
          SOLVE_TIMEOUT(1.2),
          TOTAL_TIMEOUT(5.0),
          CONV_TOL(1e-2),
          TRUST_DELTA_POS(50.0),
          TRUST_DELTA_VEL(10.0),
          VIRTUAL_CONTROL_WEIGHT(1e6),
          TERMINAL_WEIGHT(50000.0),
          SMOOTH_WEIGHT(0.1),
          Q_MAX(80000.0),
          Q_DOT_MAX(1.5e6),
          OBSTACLE_CENTER{50.0, 0.0, -50.0},
          OBSTACLE_SIZE{30.0, 30.0, 100.0},
          CD_LINEARIZE(0.3),
          REF_AREA(0.0),  // 构造时计算
          theta_max(0.6981317007977318) {}
};

// ===========================================================================
// SCvxResult - SCvx求解结果
// 对应 Python solve() 返回的 info_dict
// ===========================================================================
struct SCvxResult {
    bool         success;        // 是否成功
    bool         fallback;       // 是否降级
    bool         converged;      // 是否收敛
    int          iterations;     // prox-linear迭代次数
    scalar_t     solve_time;     // 总求解时间 [s]
    scalar_t     cost;           // 代价函数值
    SolveStatus  status;         // 求解状态
    char         solver_name[32]; // 求解器名称
    char         fail_reason[32]; // 失败原因

    int N_solved;                // 实际求解步数

    // 轨迹 (预分配, 零动态内存)
    std::array<StateVec, SCVX_N + 1> x_traj;      // 状态轨迹 [N+1, 6]
    std::array<ControlVec, SCVX_N>   u_traj;      // 控制轨迹 [N, 3]
    std::array<scalar_t, SCVX_N>     sigma_traj;  // 推力幅值轨迹 [N]
    std::array<StateVec, SCVX_N>     v_traj;      // 虚拟控制轨迹 [N, 6]

    // 重置
    void reset() {
        success   = false;
        fallback  = false;
        converged = false;
        iterations = 0;
        solve_time = 0.0;
        cost       = 0.0;
        status     = SolveStatus::OTHER_FAILURE;
        solver_name[0]  = '\0';
        fail_reason[0]  = '\0';
        N_solved = 0;
        for (std::size_t i = 0; i <= SCVX_N; ++i) {
            x_traj[i] = StateVec();
        }
        for (std::size_t i = 0; i < SCVX_N; ++i) {
            u_traj[i]     = ControlVec();
            sigma_traj[i] = 0.0;
            v_traj[i]     = StateVec();
        }
    }

    // 设置求解器名称
    void set_solver(const char* name) {
        for (int i = 0; i < 31 && name[i] != '\0'; ++i) {
            solver_name[i] = name[i];
            solver_name[i + 1] = '\0';
        }
    }

    // 设置失败原因
    void set_fail_reason(const char* reason) {
        for (int i = 0; i < 31 && reason[i] != '\0'; ++i) {
            fail_reason[i] = reason[i];
            fail_reason[i + 1] = '\0';
        }
    }
};

// ===========================================================================
// SCvxSolver - SCvx主求解器
//
// 对应 Python SCvxPlanner 类
//
// 算法流程:
//   1. 初始化参考轨迹 (warm-starting > G-FOLD > 直线插值)
//   2. 在参考轨迹上线性化气动力 (一阶Taylor展开)
//   3. 构造SOCP子问题 (信赖域 + 虚拟控制 + 燃料代价)
//   4. prox-linear迭代求解
//   5. 收敛判定: 每步平均归一化状态差 < CONV_TOL
//   6. 熔断: 8次迭代/1.2s超时/发散 → G-FOLD降级
//
// 内存布局:
//   所有矩阵和向量预分配为成员变量, 编译期固定大小.
//   总内存约 30*(6*6+6+6)*8 + 31*6*8 + 30*3*8 ≈ 15KB (栈分配)
// ===========================================================================
class SCvxSolver {
public:
    SCvxConfig config;

    // === 预分配缓冲区 (零动态内存) ===
    std::array<StateMat, SCVX_N>   A_eff_buf;     // 线性化状态矩阵 [N]
    InputMat                       B_d_buf;        // 线性化输入矩阵 (常数)
    std::array<StateVec, SCVX_N>   const_eff_buf;  // 线性化常数项 [N]

    // === 参考轨迹 (当前迭代) ===
    std::array<StateVec, SCVX_N + 1> x_bar;  // 参考状态轨迹 [N+1, 6]
    std::array<ControlVec, SCVX_N>   u_bar;  // 参考控制轨迹 [N, 3]

    // === Warm-starting (上一次SCvx成功的解) ===
    bool                             has_warm_start;
    std::array<StateVec, SCVX_N + 1> last_x_bar;
    std::array<ControlVec, SCVX_N>   last_u_bar;

    // === 统计 ===
    int solve_count;     // SCvx成功次数
    int fallback_count;  // 降级次数

    // === G-FOLD降级规划器 ===
    G_FOLD_Planner gfold;

    // -----------------------------------------------------------------------
    // 构造
    // 参数: theta_max_rad - 推力指向锥角度 [rad], 默认40°
    // -----------------------------------------------------------------------
    SCvxSolver(scalar_t theta_max_rad = 0.6981317007977318)
        : gfold(static_cast<int>(SCVX_N),
                static_cast<float>(theta_max_rad), 0.0f) {
        // 计算参考面积: π·(D/2)², D=3.35m
        config.REF_AREA = 3.14159265358979323846 * (3.35 / 2.0) * (3.35 / 2.0);
        config.theta_max = theta_max_rad;

        has_warm_start = false;
        solve_count    = 0;
        fallback_count = 0;

        // 初始化所有缓冲区为零
        for (std::size_t i = 0; i <= SCVX_N; ++i) {
            x_bar[i]      = StateVec();
            last_x_bar[i] = StateVec();
        }
        for (std::size_t i = 0; i < SCVX_N; ++i) {
            u_bar[i]         = ControlVec();
            last_u_bar[i]    = ControlVec();
            A_eff_buf[i]     = StateMat();
            const_eff_buf[i] = StateVec();
        }
        B_d_buf = InputMat();
    }

    // -----------------------------------------------------------------------
    // 主求解入口 (prox-linear迭代 + 熔断)
    //
    // 对应 Python SCvxPlanner.solve
    //
    // 参数:
    //   pos0[3]:       当前位置 (NED, m)
    //   vel0[3]:       当前速度 (NED, m/s)
    //   pos_target[3]: 终端目标位置 (NED, m)
    //   vel_target[3]: 终端目标速度 (NED, m/s)
    //   m_current:     当前质量 [kg]
    //   tgo:           剩余时间 [s]
    //   F_aero_n[3]:   已知气动力 (N, 兼容G-FOLD接口, 仅降级时使用)
    //   wind_n[3]:     风速 (NED, m/s, 用于气动力线性化)
    //   n_engines:     发动机数量 (1或3)
    //   multi_constraint: 是否启用多约束 (动压/热流/避障)
    //
    // 返回: SCvxResult (含轨迹和状态信息)
    // -----------------------------------------------------------------------
    SCvxResult solve(const scalar_t pos0[3], const scalar_t vel0[3],
                     const scalar_t pos_target[3], const scalar_t vel_target[3],
                     scalar_t m_current, scalar_t tgo,
                     const scalar_t F_aero_n[3] = nullptr,
                     const scalar_t wind_n[3]   = nullptr,
                     int  n_engines        = 1,
                     bool multi_constraint = false);

    // -----------------------------------------------------------------------
    // 计算气动力加速度及其Jacobian w.r.t. 速度
    //
    // 对应 Python _compute_aero_jacobian
    //
    // 简化模型 (drag-only):
    //   F_aero = -0.5 * ρ * Cd * S * |V_rel| * V_rel
    //   a_aero = F_aero / m
    //
    // Jacobian:
    //   da_aero/dV = -k/m * (V_r·V_r^T/|V_r| + |V_r|·I)
    //   where k = 0.5 * ρ * Cd * S
    //
    // 参数:
    //   vel_n:  NED速度 [m/s]
    //   wind_n: NED风速 [m/s]
    //   h:      高度 [m]
    //   m:      质量 [kg]
    // 输出:
    //   a_aero:  气动力加速度 [m/s²]
    //   J_aero:  气动力加速度对速度的Jacobian [3,3]
    // -----------------------------------------------------------------------
    void compute_aero_jacobian(const Vec3d& vel_n, const Vec3d& wind_n,
                               scalar_t h, scalar_t m,
                               Vec3d& a_aero, Mat3d& J_aero);

    // -----------------------------------------------------------------------
    // 在参考轨迹上线性化动力学
    //
    // 对应 Python _linearize_dynamics
    //
    // 非线性动力学:
    //   pos[k+1] = pos[k] + vel[k]*dt + 0.5*(T[k]/m + g + a_aero(V[k]))*dt²
    //   vel[k+1] = vel[k] + (T[k]/m + g + a_aero(V[k]))*dt
    //
    // 线性化 (一阶Taylor展开 a_aero(V) ≈ a_bar + J@(V - V_bar)):
    //   x[k+1] = A_eff[k] @ x[k] + B_d @ u[k] + const_eff[k]
    //
    // 参数:
    //   x_bar_in:  参考状态轨迹 [N+1, 6]
    //   u_bar_in:  参考控制轨迹 [N, 3] (未直接使用, 保留接口)
    //   wind_n:    NED风速
    //   m:         质量
    //   dt:        步长
    //   N_actual:  实际步数
    // 输出:
    //   A_eff_list:    线性化状态矩阵 [N]
    //   B_d:           线性化输入矩阵 (常数)
    //   const_eff_list: 线性化常数项 [N]
    // -----------------------------------------------------------------------
    void linearize_dynamics(const std::array<StateVec, SCVX_N + 1>& x_bar_in,
                            const std::array<ControlVec, SCVX_N>& u_bar_in,
                            const Vec3d& wind_n, scalar_t m, scalar_t dt,
                            int N_actual,
                            std::array<StateMat, SCVX_N>& A_eff_list,
                            InputMat& B_d,
                            std::array<StateVec, SCVX_N>& const_eff_list);

    // -----------------------------------------------------------------------
    // 求解SCvx SOCP子问题
    //
    // 对应 Python _solve_scvx_subproblem
    //
    // 如果定义了SCVX_USE_OSQP且OSQP可用, 使用OSQP求解;
    // 否则降级到解析求解 (基于能量守恒的近似解).
    //
    // 参数:
    //   x0, x_target:     初始/终端状态
    //   A_eff_list, B_d, const_eff_list: 线性化动力学
    //   x_bar_in, u_bar_in: 参考轨迹 (信赖域中心)
    //   m, dt, T_max, T_min: 物理参数
    //   N_actual:         实际步数
    //   multi_constraint: 是否启用多约束
    // -----------------------------------------------------------------------
    SCvxResult solve_scvx_subproblem(
        const StateVec& x0, const StateVec& x_target,
        const std::array<StateMat, SCVX_N>& A_eff_list,
        const InputMat& B_d,
        const std::array<StateVec, SCVX_N>& const_eff_list,
        const std::array<StateVec, SCVX_N + 1>& x_bar_in,
        const std::array<ControlVec, SCVX_N>& u_bar_in,
        scalar_t m, scalar_t dt,
        scalar_t T_max, scalar_t T_min,
        int N_actual, bool multi_constraint);

    // -----------------------------------------------------------------------
    // 用简化动力学仿真轨迹 (用于warm-starting)
    //
    // 对应 Python _simulate_trajectory
    //
    // 动力学: x[k+1] = A_d@x[k] + B_d@u[k] + g_d + aero_d(x[k])
    // (气动力零阶保持, 与G-FOLD动力学一致)
    //
    // 参数:
    //   x0:        初始状态
    //   u_traj_in: 控制轨迹 [N, 3]
    //   wind_n:    风速
    //   m, dt:     质量, 步长
    //   N_actual:  实际步数
    // 输出:
    //   x_traj_out: 仿真状态轨迹 [N+1, 6]
    // -----------------------------------------------------------------------
    void simulate_trajectory(const StateVec& x0,
                             const std::array<ControlVec, SCVX_N>& u_traj_in,
                             const Vec3d& wind_n, scalar_t m, scalar_t dt,
                             int N_actual,
                             std::array<StateVec, SCVX_N + 1>& x_traj_out);

    // -----------------------------------------------------------------------
    // 重置统计和warm-starting状态
    // -----------------------------------------------------------------------
    void reset() {
        has_warm_start = false;
        solve_count    = 0;
        fallback_count = 0;
        gfold.reset();
    }

    // -----------------------------------------------------------------------
    // 获取统计信息
    // -----------------------------------------------------------------------
    void get_stats(int& total_solves, int& scvx_success, int& fallback_cnt,
                   scalar_t& fallback_rate) const {
        total_solves   = solve_count + fallback_count;
        scvx_success   = solve_count;
        fallback_cnt   = fallback_count;
        fallback_rate  = (total_solves > 0)
                         ? static_cast<scalar_t>(fallback_count) / total_solves
                         : 0.0;
    }

private:
    // -----------------------------------------------------------------------
    // OSQP接口 (条件编译)
    // 当定义SCVX_USE_OSQP时使用OSQP求解SOCP子问题
    // -----------------------------------------------------------------------
#ifdef SCVX_USE_OSQP
    SCvxResult solve_with_osqp(
        const StateVec& x0, const StateVec& x_target,
        const std::array<StateMat, SCVX_N>& A_eff_list,
        const InputMat& B_d,
        const std::array<StateVec, SCVX_N>& const_eff_list,
        const std::array<StateVec, SCVX_N + 1>& x_bar_in,
        const std::array<ControlVec, SCVX_N>& u_bar_in,
        scalar_t m, scalar_t dt,
        scalar_t T_max, scalar_t T_min,
        int N_actual, bool multi_constraint);
#endif

    // -----------------------------------------------------------------------
    // 解析求解 (OSQP不可用时的降级方案)
    //
    // 基于能量守恒的近似解, 与G-FOLD解析fallback类似:
    //   1. 计算到达目标所需的加速度
    //   2. 推力: T = m*(a - g - a_aero)
    //   3. 幅值限幅 [T_min, T_max], 指向锥约束
    //   4. 信赖域约束
    //   5. 多约束检查 (动压/热流/避障)
    // -----------------------------------------------------------------------
    SCvxResult solve_analytical(
        const StateVec& x0, const StateVec& x_target,
        const std::array<StateMat, SCVX_N>& A_eff_list,
        const InputMat& B_d,
        const std::array<StateVec, SCVX_N>& const_eff_list,
        const std::array<StateVec, SCVX_N + 1>& x_bar_in,
        const std::array<ControlVec, SCVX_N>& u_bar_in,
        scalar_t m, scalar_t dt,
        scalar_t T_max, scalar_t T_min,
        int N_actual, bool multi_constraint);

    // -----------------------------------------------------------------------
    // 障碍物绕行: 修改参考轨迹, 避免穿过障碍物
    //
    // 对应 Python solve() 中的障碍物穿透检测逻辑
    //
    // 策略: 检测参考轨迹是否穿过障碍物xy投影, 若穿透则施加sin曲线y偏移
    // -----------------------------------------------------------------------
    void apply_obstacle_detour(std::array<StateVec, SCVX_N + 1>& x_bar_in,
                               int N_actual);

    // -----------------------------------------------------------------------
    // 检查参考轨迹是否穿过障碍物
    // -----------------------------------------------------------------------
    bool check_obstacle_penetration(
        const std::array<StateVec, SCVX_N + 1>& x_bar_in, int N_actual);

    // -----------------------------------------------------------------------
    // 应用多约束修正 (动压/热流/避障)
    // 在解析求解后对控制轨迹进行后验修正
    // -----------------------------------------------------------------------
    void apply_multi_constraint_correction(
        std::array<StateVec, SCVX_N + 1>& x_traj_out,
        std::array<ControlVec, SCVX_N>& u_traj_out,
        const std::array<StateVec, SCVX_N + 1>& x_bar_in,
        int N_actual, scalar_t m, scalar_t dt);
};

// ===========================================================================
// 方法实现
// ===========================================================================

// -----------------------------------------------------------------------
// 计算气动力加速度及其Jacobian
// -----------------------------------------------------------------------
void SCvxSolver::compute_aero_jacobian(const Vec3d& vel_n, const Vec3d& wind_n,
                                       scalar_t h, scalar_t m,
                                       Vec3d& a_aero, Mat3d& J_aero) {
    // 大气密度
    scalar_t rho = scvx_atmos::density(h);

    // 低速或稀薄大气: 气动力可忽略, Jacobian=0 (退化为G-FOLD)
    Vec3d V_r = vel_n - wind_n;
    scalar_t V_r_mag = V_r.norm();

    if (V_r_mag < 0.5 || rho < scvx_atmos::RHO_THRESHOLD) {
        a_aero = Vec3d();
        J_aero = Mat3d();
        return;
    }

    // 阻力系数: k = 0.5 * ρ * Cd * S
    scalar_t k = 0.5 * rho * config.CD_LINEARIZE * config.REF_AREA;

    // 气动力加速度 (NED, 阻力沿-V_rel方向)
    // a_aero = -k/m * |V_rel| * V_rel
    scalar_t coef = -k / m * V_r_mag;
    a_aero[0] = coef * V_r[0];
    a_aero[1] = coef * V_r[1];
    a_aero[2] = coef * V_r[2];

    // Jacobian w.r.t. V
    // da/dV = -k/m * (V_r·V_r^T/|V_r| + |V_r|·I)
    scalar_t inv_Vr = 1.0 / V_r_mag;
    scalar_t k_over_m = k / m;

    // J = -k/m * (outer(V_r, V_r)/|V_r| + |V_r|*I)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            scalar_t outer_term = V_r[i] * V_r[j] * inv_Vr;
            scalar_t identity_term = (i == j) ? V_r_mag : 0.0;
            J_aero(i, j) = -k_over_m * (outer_term + identity_term);
        }
    }
}

// -----------------------------------------------------------------------
// 在参考轨迹上线性化动力学
// -----------------------------------------------------------------------
void SCvxSolver::linearize_dynamics(
    const std::array<StateVec, SCVX_N + 1>& x_bar_in,
    const std::array<ControlVec, SCVX_N>& u_bar_in,
    const Vec3d& wind_n, scalar_t m, scalar_t dt,
    int N_actual,
    std::array<StateMat, SCVX_N>& A_eff_list,
    InputMat& B_d,
    std::array<StateVec, SCVX_N>& const_eff_list) {
    (void)u_bar_in;  // 未直接使用, 保留接口

    // I3, Z3
    Mat3d I3 = Mat3d::Identity();
    Mat3d Z3;  // 零矩阵 (默认构造为零)

    // B_d 是常数 (不依赖参考轨迹)
    // B_d = [[0.5*dt²/m * I3],
    //        [dt/m * I3]]
    scalar_t dt2 = dt * dt;
    scalar_t B_pos_coef = 0.5 * dt2 / m;
    scalar_t B_vel_coef = dt / m;

    B_d = InputMat();  // 清零
    for (int i = 0; i < 3; ++i) {
        B_d(i, i)     = B_pos_coef;  // 位置行
        B_d(i + 3, i) = B_vel_coef;  // 速度行
    }

    // 重力向量 (NED, +Z向下)
    Vec3d g_vec;
    g_vec[0] = 0.0;
    g_vec[1] = 0.0;
    g_vec[2] = scvx_atmos::G0;

    // 对每一步线性化
    for (int k = 0; k < N_actual && k < static_cast<int>(SCVX_N); ++k) {
        // 提取参考轨迹点
        Vec3d pos_bar, vel_bar;
        pos_bar[0] = x_bar_in[k][0];
        pos_bar[1] = x_bar_in[k][1];
        pos_bar[2] = x_bar_in[k][2];
        vel_bar[0] = x_bar_in[k][3];
        vel_bar[1] = x_bar_in[k][4];
        vel_bar[2] = x_bar_in[k][5];

        scalar_t h_bar = -pos_bar[2];  // 高度 = -Z (NED)

        // 计算气动力加速度和Jacobian
        Vec3d a_aero_bar;
        Mat3d J_aero_v;
        compute_aero_jacobian(vel_bar, wind_n, h_bar, m, a_aero_bar, J_aero_v);

        // A_eff = [[I3, dt*I3 + 0.5*dt²*J],
        //          [Z3, I3 + dt*J]]
        Mat3d A11 = I3;
        Mat3d A12 = I3 * dt + J_aero_v * (0.5 * dt2);
        Mat3d A21 = Z3;
        Mat3d A22 = I3 + J_aero_v * dt;

        A_eff_list[k] = StateMat();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A_eff_list[k](i, j)         = A11(i, j);  // 左上
                A_eff_list[k](i, j + 3)     = A12(i, j);  // 右上
                A_eff_list[k](i + 3, j)     = A21(i, j);  // 左下
                A_eff_list[k](i + 3, j + 3) = A22(i, j);  // 右下
            }
        }

        // const_eff = [0.5*dt²*(g + a_bar - J@V_bar),
        //              dt*(g + a_bar - J@V_bar)]
        // (J@V_bar项是Taylor展开的修正, 使线性化在V_bar处精确)
        Vec3d J_V_bar = J_aero_v * vel_bar;
        Vec3d residual = g_vec + a_aero_bar - J_V_bar;

        const_eff_list[k] = StateVec();
        for (int i = 0; i < 3; ++i) {
            const_eff_list[k][i]     = 0.5 * dt2 * residual[i];  // 位置行
            const_eff_list[k][i + 3] = dt * residual[i];          // 速度行
        }
    }
}

// -----------------------------------------------------------------------
// 求解SCvx SOCP子问题
// -----------------------------------------------------------------------
SCvxResult SCvxSolver::solve_scvx_subproblem(
    const StateVec& x0, const StateVec& x_target,
    const std::array<StateMat, SCVX_N>& A_eff_list,
    const InputMat& B_d,
    const std::array<StateVec, SCVX_N>& const_eff_list,
    const std::array<StateVec, SCVX_N + 1>& x_bar_in,
    const std::array<ControlVec, SCVX_N>& u_bar_in,
    scalar_t m, scalar_t dt,
    scalar_t T_max, scalar_t T_min,
    int N_actual, bool multi_constraint) {

#ifdef SCVX_USE_OSQP
    // OSQP可用时使用OSQP求解
    return solve_with_osqp(x0, x_target, A_eff_list, B_d, const_eff_list,
                           x_bar_in, u_bar_in, m, dt, T_max, T_min,
                           N_actual, multi_constraint);
#else
    // OSQP不可用: 降级到解析求解
    return solve_analytical(x0, x_target, A_eff_list, B_d, const_eff_list,
                            x_bar_in, u_bar_in, m, dt, T_max, T_min,
                            N_actual, multi_constraint);
#endif
}

// -----------------------------------------------------------------------
// 解析求解 (OSQP不可用时的降级方案)
// -----------------------------------------------------------------------
SCvxResult SCvxSolver::solve_analytical(
    const StateVec& x0, const StateVec& x_target,
    const std::array<StateMat, SCVX_N>& A_eff_list,
    const InputMat& B_d,
    const std::array<StateVec, SCVX_N>& const_eff_list,
    const std::array<StateVec, SCVX_N + 1>& x_bar_in,
    const std::array<ControlVec, SCVX_N>& u_bar_in,
    scalar_t m, scalar_t dt,
    scalar_t T_max, scalar_t T_min,
    int N_actual, bool multi_constraint) {
    (void)u_bar_in;  // 参考控制未直接使用, 保留接口

    SCvxResult result;
    result.reset();
    result.N_solved = N_actual;

    // === 归一化 (与G-FOLD一致, 保证数值稳定性) ===
    const scalar_t s_p = 100.0;  // 位置归一化
    const scalar_t s_v = 10.0;   // 速度归一化

    // === 解析推力剖面 (基于位置约束 + SOCP软约束对齐) ===
    // 匀加速运动: pos(tgo) = pos0 + vel0*tgo + 0.5*a*tgo²
    // => a = 2*(pos_target - pos0 - vel0*tgo) / tgo²
    scalar_t tgo = dt * N_actual;
    scalar_t tgo2 = tgo * tgo;

    // 位置误差
    scalar_t e_pos[3];
    for (int i = 0; i < 3; ++i) {
        e_pos[i] = x_target[i] - x0[i] - x0[i + 3] * tgo;
    }

    // 水平位置软约束 (20m, 与Python SOCP对齐)
    const scalar_t SOFT_POS_TOL = 20.0;
    scalar_t e_pos_xy_mag = std::sqrt(e_pos[0] * e_pos[0] + e_pos[1] * e_pos[1]);
    scalar_t xy_pos_scale = 1.0;
    if (e_pos_xy_mag > SOFT_POS_TOL) {
        xy_pos_scale = (e_pos_xy_mag - SOFT_POS_TOL) / e_pos_xy_mag;
    } else {
        xy_pos_scale = 0.0;  // 已在软约束球内
    }

    // 水平速度软约束 (10m/s)
    const scalar_t SOFT_VEL_TOL = 10.0;
    scalar_t e_vel_xy[2] = {x_target[3] - x0[3], x_target[4] - x0[4]};
    scalar_t e_vel_xy_mag = std::sqrt(e_vel_xy[0] * e_vel_xy[0] +
                                      e_vel_xy[1] * e_vel_xy[1]);
    scalar_t xy_vel_scale = 0.0;
    if (e_vel_xy_mag > SOFT_VEL_TOL) {
        xy_vel_scale = (e_vel_xy_mag - SOFT_VEL_TOL) / e_vel_xy_mag;
    }

    // 所需加速度
    // Phase 5B修复: Z轴优先终端速度约束 (着陆安全)
    // 原代码: a_req[2] = 2.0 * e_pos[2] / tgo2  只考虑位置, 末端速度偏差大
    // 修复: a = a_vel + k * pos_residual / tgo²
    //   a_vel = (vel_target - vel0) / tgo  (速度约束, 精确满足末端速度)
    //   pos_residual = e_pos - 0.5*a_vel*tgo²  (速度约束下的位置残差)
    //   k = 0.3 (位置修正增益, <1以优先速度, MPC反馈修正位置)
    scalar_t a_req[3];
    a_req[0] = 2.0 * e_pos[0] * xy_pos_scale / tgo2 + e_vel_xy[0] * xy_vel_scale / tgo;
    a_req[1] = 2.0 * e_pos[1] * xy_pos_scale / tgo2 + e_vel_xy[1] * xy_vel_scale / tgo;

    // Z轴: 速度约束优先 + 位置修正
    scalar_t a_vel_z = (x_target[5] - x0[5]) / tgo;  // 速度约束加速度
    scalar_t pos_residual_z = e_pos[2] - 0.5 * a_vel_z * tgo2;  // 速度约束下位置残差
    a_req[2] = a_vel_z + 0.3 * pos_residual_z / tgo2;  // 速度优先, 位置修正

    // 推力: T = m*(a - g - F_aero/m)
    // g_n = [0, 0, G0] (NED, +Z向下)
    // 气动力使用参考轨迹上的零阶保持近似
    Vec3d wind_zero;  // 默认零风
    wind_zero[0] = wind_zero[1] = wind_zero[2] = 0.0;

    // 计算初始位置的气动力 (零阶保持)
    Vec3d vel0_vec, a_aero0;
    Mat3d J_dummy;
    vel0_vec[0] = x0[3]; vel0_vec[1] = x0[4]; vel0_vec[2] = x0[5];
    scalar_t h0 = -x0[2];
    compute_aero_jacobian(vel0_vec, wind_zero, h0, m, a_aero0, J_dummy);

    scalar_t T_req[3];
    T_req[0] = m * a_req[0] - m * a_aero0[0];
    T_req[1] = m * a_req[1] - m * a_aero0[1];
    T_req[2] = m * (a_req[2] - scvx_atmos::G0) - m * a_aero0[2];

    // NaN检查
    scalar_t T_mag = std::sqrt(T_req[0] * T_req[0] +
                               T_req[1] * T_req[1] +
                               T_req[2] * T_req[2]);
    if (T_mag != T_mag) {  // NaN
        result.status = SolveStatus::OTHER_FAILURE;
        result.set_fail_reason("NaN_thrust");
        return result;
    }

    // 幅值限幅 [T_min, T_max]
    if (T_mag < 1.0) {
        T_req[0] = 0.0; T_req[1] = 0.0; T_req[2] = -T_min;
        T_mag = T_min;
    } else if (T_mag > T_max) {
        scalar_t scale = T_max / T_mag;
        T_req[0] *= scale; T_req[1] *= scale; T_req[2] *= scale;
        T_mag = T_max;
    }

    // 指向锥约束: |T_xy| <= tan(theta) * (-T_z)
    if (T_req[2] >= 0.0) {
        // 推力不向上! 强制垂直向上用 T_min
        T_req[0] = 0.0; T_req[1] = 0.0; T_req[2] = -T_min;
        T_mag = T_min;
    } else {
        scalar_t T_xy_mag = std::sqrt(T_req[0] * T_req[0] + T_req[1] * T_req[1]);
        scalar_t tan_theta = std::tan(config.theta_max);
        scalar_t max_xy = tan_theta * (-T_req[2]);
        if (T_xy_mag > max_xy && max_xy > 0.0) {
            scalar_t scale = max_xy / T_xy_mag;
            T_req[0] *= scale;
            T_req[1] *= scale;
            T_mag = std::sqrt(T_req[0] * T_req[0] + T_req[1] * T_req[1] +
                              T_req[2] * T_req[2]);
        }
    }

    // === 填充轨迹 (零阶保持) ===
    for (int k = 0; k < N_actual && k < static_cast<int>(SCVX_N); ++k) {
        result.u_traj[k][0] = T_req[0];
        result.u_traj[k][1] = T_req[1];
        result.u_traj[k][2] = T_req[2];
        result.sigma_traj[k] = T_mag;
        result.v_traj[k] = StateVec();  // 虚拟控制=0 (解析解无需虚拟控制)
    }

    // === 仿真状态轨迹 (使用线性化动力学) ===
    result.x_traj[0] = x0;
    for (int k = 0; k < N_actual && k < static_cast<int>(SCVX_N); ++k) {
        // x[k+1] = A_eff[k] @ x[k] + B_d @ u[k] + const_eff[k]
        StateVec Ax = A_eff_list[k] * result.x_traj[k];
        ControlVec u_k;
        u_k[0] = result.u_traj[k][0];
        u_k[1] = result.u_traj[k][1];
        u_k[2] = result.u_traj[k][2];
        StateVec Bu = B_d * u_k;
        result.x_traj[k + 1] = Ax + Bu + const_eff_list[k];
    }

    // === 信赖域约束 (解析修正: 将状态限制在信赖域内) ===
    for (int k = 0; k <= N_actual && k < static_cast<int>(SCVX_N + 1); ++k) {
        for (int i = 0; i < 3; ++i) {
            // 位置信赖域
            scalar_t delta_pos = result.x_traj[k][i] - x_bar_in[k][i];
            scalar_t delta_pos_norm = delta_pos / s_p;
            if (std::abs(delta_pos_norm) > config.TRUST_DELTA_POS / s_p) {
                scalar_t sign = (delta_pos > 0) ? 1.0 : -1.0;
                result.x_traj[k][i] = x_bar_in[k][i] +
                    sign * config.TRUST_DELTA_POS;
            }
            // 速度信赖域
            scalar_t delta_vel = result.x_traj[k][i + 3] - x_bar_in[k][i + 3];
            scalar_t delta_vel_norm = delta_vel / s_v;
            if (std::abs(delta_vel_norm) > config.TRUST_DELTA_VEL / s_v) {
                scalar_t sign = (delta_vel > 0) ? 1.0 : -1.0;
                result.x_traj[k][i + 3] = x_bar_in[k][i + 3] +
                    sign * config.TRUST_DELTA_VEL;
            }
        }
    }

    // === 多约束修正 ===
    if (multi_constraint) {
        apply_multi_constraint_correction(result.x_traj, result.u_traj,
                                          x_bar_in, N_actual, m, dt);
    }

    // === 代价 (近似: sum(sigma)) ===
    result.cost = 0.0;
    for (int k = 0; k < N_actual && k < static_cast<int>(SCVX_N); ++k) {
        result.cost += result.sigma_traj[k];
    }

    result.success   = true;
    result.converged = false;  // 由外层prox-linear判定
    result.status    = SolveStatus::OPTIMAL;
    result.set_solver("ANALYTICAL");
    return result;
}

// -----------------------------------------------------------------------
// 用简化动力学仿真轨迹
// -----------------------------------------------------------------------
void SCvxSolver::simulate_trajectory(
    const StateVec& x0,
    const std::array<ControlVec, SCVX_N>& u_traj_in,
    const Vec3d& wind_n, scalar_t m, scalar_t dt,
    int N_actual,
    std::array<StateVec, SCVX_N + 1>& x_traj_out) {

    x_traj_out[0] = x0;
    scalar_t dt2 = dt * dt;

    for (int k = 0; k < N_actual && k < static_cast<int>(SCVX_N); ++k) {
        // 提取当前状态
        Vec3d pos, vel;
        pos[0] = x_traj_out[k][0]; pos[1] = x_traj_out[k][1]; pos[2] = x_traj_out[k][2];
        vel[0] = x_traj_out[k][3]; vel[1] = x_traj_out[k][4]; vel[2] = x_traj_out[k][5];
        scalar_t h = -pos[2];

        // 气动力 (在当前状态计算)
        Vec3d a_aero;
        Mat3d J_dummy;
        compute_aero_jacobian(vel, wind_n, h, m, a_aero, J_dummy);

        // 简化动力学 (与G-FOLD一致)
        // acc = T/m + g + a_aero
        // g = [0, 0, G0]
        scalar_t acc[3];
        acc[0] = u_traj_in[k][0] / m + a_aero[0];
        acc[1] = u_traj_in[k][1] / m + a_aero[1];
        acc[2] = u_traj_in[k][2] / m + scvx_atmos::G0 + a_aero[2];

        // pos[k+1] = pos + vel*dt + 0.5*acc*dt²
        // vel[k+1] = vel + acc*dt
        x_traj_out[k + 1][0] = pos[0] + vel[0] * dt + 0.5 * acc[0] * dt2;
        x_traj_out[k + 1][1] = pos[1] + vel[1] * dt + 0.5 * acc[1] * dt2;
        x_traj_out[k + 1][2] = pos[2] + vel[2] * dt + 0.5 * acc[2] * dt2;
        x_traj_out[k + 1][3] = vel[0] + acc[0] * dt;
        x_traj_out[k + 1][4] = vel[1] + acc[1] * dt;
        x_traj_out[k + 1][5] = vel[2] + acc[2] * dt;
    }
}

// -----------------------------------------------------------------------
// 检查参考轨迹是否穿过障碍物
// -----------------------------------------------------------------------
bool SCvxSolver::check_obstacle_penetration(
    const std::array<StateVec, SCVX_N + 1>& x_bar_in, int N_actual) {

    scalar_t obs_cx = config.OBSTACLE_CENTER[0];
    scalar_t obs_cy = config.OBSTACLE_CENTER[1];
    scalar_t obs_hx = config.OBSTACLE_SIZE[0] / 2.0;
    scalar_t obs_hy = config.OBSTACLE_SIZE[1] / 2.0;

    for (int k = 0; k <= N_actual && k < static_cast<int>(SCVX_N + 1); ++k) {
        scalar_t px = x_bar_in[k][0];
        scalar_t py = x_bar_in[k][1];
        if (std::abs(px - obs_cx) < obs_hx && std::abs(py - obs_cy) < obs_hy) {
            return true;  // 穿透
        }
    }
    return false;
}

// -----------------------------------------------------------------------
// 障碍物绕行: 修改参考轨迹
// -----------------------------------------------------------------------
void SCvxSolver::apply_obstacle_detour(
    std::array<StateVec, SCVX_N + 1>& x_bar_in, int N_actual) {

    if (!check_obstacle_penetration(x_bar_in, N_actual)) {
        return;  // 无穿透, 无需绕行
    }

    // 生成y方向绕行偏移: sin曲线, 中间最大, 两端为0
    // 绕行裕度: 半尺寸 + 20m
    scalar_t y_offset = config.OBSTACLE_SIZE[1] / 2.0 + 20.0;

    for (int k = 0; k <= N_actual && k < static_cast<int>(SCVX_N + 1); ++k) {
        scalar_t sin_profile = std::sin(3.14159265358979323846 * k / N_actual);
        x_bar_in[k][1] += y_offset * sin_profile;
    }
}

// -----------------------------------------------------------------------
// 应用多约束修正 (动压/热流/避障)
// -----------------------------------------------------------------------
void SCvxSolver::apply_multi_constraint_correction(
    std::array<StateVec, SCVX_N + 1>& x_traj_out,
    std::array<ControlVec, SCVX_N>& u_traj_out,
    const std::array<StateVec, SCVX_N + 1>& x_bar_in,
    int N_actual, scalar_t m, scalar_t dt) {
    (void)dt;  // dt在逆动力学中使用

    // 障碍物参数
    scalar_t obs_cx = config.OBSTACLE_CENTER[0];
    scalar_t obs_cy = config.OBSTACLE_CENTER[1];
    scalar_t obs_hx = config.OBSTACLE_SIZE[0] / 2.0;
    scalar_t obs_hy = config.OBSTACLE_SIZE[1] / 2.0;

    for (int k = 0; k <= N_actual && k < static_cast<int>(SCVX_N + 1); ++k) {
        // 提取状态
        scalar_t px = x_traj_out[k][0];
        scalar_t py = x_traj_out[k][1];
        scalar_t pz = x_traj_out[k][2];
        scalar_t vx = x_traj_out[k][3];
        scalar_t vy = x_traj_out[k][4];
        scalar_t vz = x_traj_out[k][5];

        scalar_t h = -pz;
        scalar_t v_mag = std::sqrt(vx * vx + vy * vy + vz * vz);
        scalar_t v_mag2 = v_mag * v_mag;

        // === 动压约束: ρ(h) * ||v||² <= Q_MAX ===
        scalar_t rho = scvx_atmos::density(h);
        if (rho > scvx_atmos::RHO_THRESHOLD && v_mag2 > 1.0) {
            scalar_t q = rho * v_mag2;
            if (q > config.Q_MAX) {
                // 速度过大, 按比例缩放 (后验修正)
                scalar_t scale = std::sqrt(config.Q_MAX / q);
                x_traj_out[k][3] = vx * scale;
                x_traj_out[k][4] = vy * scale;
                x_traj_out[k][5] = vz * scale;
            }
        }

        // === 热流约束: ρ(h) * ||v||³ <= Q_DOT_MAX ===
        if (rho > scvx_atmos::RHO_THRESHOLD && v_mag2 > 1.0) {
            scalar_t q_dot = rho * v_mag2 * v_mag;
            if (q_dot > config.Q_DOT_MAX) {
                scalar_t scale = std::pow(config.Q_DOT_MAX / q_dot, 1.0 / 3.0);
                x_traj_out[k][3] *= scale;
                x_traj_out[k][4] *= scale;
                x_traj_out[k][5] *= scale;
            }
        }

        // === 避障约束: 多面体half-space ===
        // 检查是否在障碍物xy投影内
        scalar_t dx = px - obs_cx;
        scalar_t dy = py - obs_cy;
        bool inside_x = std::abs(dx) < obs_hx;
        bool inside_y = std::abs(dy) < obs_hy;

        if (inside_x && inside_y) {
            // 穿入障碍物: 强制y方向绕行
            // 选择y方向 (与Python逻辑一致)
            if (dy >= 0) {
                // 强制y >= obs_cy + obs_hy
                x_traj_out[k][1] = obs_cy + obs_hy + 1.0;
            } else {
                // 强制y <= obs_cy - obs_hy
                x_traj_out[k][1] = obs_cy - obs_hy - 1.0;
            }
        }

        // 信赖域约束 (确保修正后的状态不偏离参考太远)
        for (int i = 0; i < 3; ++i) {
            scalar_t delta_pos = x_traj_out[k][i] - x_bar_in[k][i];
            if (std::abs(delta_pos) > config.TRUST_DELTA_POS) {
                scalar_t sign = (delta_pos > 0) ? 1.0 : -1.0;
                x_traj_out[k][i] = x_bar_in[k][i] + sign * config.TRUST_DELTA_POS;
            }
            scalar_t delta_vel = x_traj_out[k][i + 3] - x_bar_in[k][i + 3];
            if (std::abs(delta_vel) > config.TRUST_DELTA_VEL) {
                scalar_t sign = (delta_vel > 0) ? 1.0 : -1.0;
                x_traj_out[k][i + 3] = x_bar_in[k][i + 3] + sign * config.TRUST_DELTA_VEL;
            }
        }
    }

    // 根据修正后的状态轨迹重新计算控制轨迹 (逆动力学)
    // T = m * (a - g - a_aero)
    // a = (vel[k+1] - vel[k]) / dt
    Vec3d wind_zero;
    wind_zero[0] = wind_zero[1] = wind_zero[2] = 0.0;

    for (int k = 0; k < N_actual && k < static_cast<int>(SCVX_N); ++k) {
        Vec3d vel_k, vel_k1;
        vel_k[0] = x_traj_out[k][3]; vel_k[1] = x_traj_out[k][4]; vel_k[2] = x_traj_out[k][5];
        vel_k1[0] = x_traj_out[k + 1][3]; vel_k1[1] = x_traj_out[k + 1][4]; vel_k1[2] = x_traj_out[k + 1][5];

        // 加速度
        Vec3d acc = (vel_k1 - vel_k) * (1.0 / dt);

        // 气动力
        Vec3d pos_k;
        pos_k[0] = x_traj_out[k][0]; pos_k[1] = x_traj_out[k][1]; pos_k[2] = x_traj_out[k][2];
        scalar_t h_k = -pos_k[2];
        Vec3d a_aero;
        Mat3d J_dummy;
        compute_aero_jacobian(vel_k, wind_zero, h_k, m, a_aero, J_dummy);

        // 推力
        u_traj_out[k][0] = m * (acc[0] - a_aero[0]);
        u_traj_out[k][1] = m * (acc[1] - a_aero[1]);
        u_traj_out[k][2] = m * (acc[2] - scvx_atmos::G0 - a_aero[2]);

        // 幅值
        scalar_t T_mag = std::sqrt(u_traj_out[k][0] * u_traj_out[k][0] +
                                   u_traj_out[k][1] * u_traj_out[k][1] +
                                   u_traj_out[k][2] * u_traj_out[k][2]);
        (void)T_mag;  // 重新计算sigma时使用 (当前版本保留原sigma)
    }
}

// -----------------------------------------------------------------------
// 主求解入口
// -----------------------------------------------------------------------
SCvxResult SCvxSolver::solve(const scalar_t pos0[3], const scalar_t vel0[3],
                             const scalar_t pos_target[3], const scalar_t vel_target[3],
                             scalar_t m_current, scalar_t tgo,
                             const scalar_t F_aero_n[3],
                             const scalar_t wind_n[3],
                             int n_engines,
                             bool multi_constraint) {
    SCvxResult result;
    result.reset();

    // === 输入检查 ===
    if (tgo < 0.1 || m_current < 1.0) {
        result.status = SolveStatus::OTHER_FAILURE;
        result.set_fail_reason("invalid_input");
        return result;
    }

    // === dt对齐 (与G-FOLD相同策略) ===
    // dt=max(1.0, tgo/N), N=tgo/dt, 保证N*dt=tgo
    scalar_t dt = std::max(1.0, tgo / static_cast<scalar_t>(SCVX_N));
    int N_actual = static_cast<int>(std::round(tgo / dt));
    if (N_actual < 1) N_actual = 1;
    if (N_actual > static_cast<int>(SCVX_N)) N_actual = static_cast<int>(SCVX_N);
    dt = tgo / N_actual;
    result.N_solved = N_actual;

    // === 推力限幅 (与G-FOLD一致) ===
    scalar_t h_current    = -pos0[2];
    scalar_t T_max_single = rocket_params::thrust_at_alt(static_cast<float>(h_current));
    scalar_t T_max        = n_engines * T_max_single;
    scalar_t T_min        = n_engines * guidance_params::T_MIN_FRAC * T_max_single;

    // === 状态向量 ===
    StateVec x0, x_target;
    for (int i = 0; i < 3; ++i) {
        x0[i]      = pos0[i];
        x0[i + 3]  = vel0[i];
        x_target[i]     = pos_target[i];
        x_target[i + 3] = vel_target[i];
    }

    // === 风速 ===
    Vec3d wind;
    if (wind_n) {
        wind[0] = wind_n[0]; wind[1] = wind_n[1]; wind[2] = wind_n[2];
    } else {
        wind[0] = wind[1] = wind[2] = 0.0;
    }

    // === 初始化参考轨迹 ===
    // 优先级: warm-starting > G-FOLD > 直线插值
    bool init_ok = false;

    if (has_warm_start) {
        // 用上一次SCvx成功的解作为初始参考 (起点终点双对齐)
        x_bar = last_x_bar;
        u_bar = last_u_bar;
        // 空间平移: 起点对齐x0, 终点对齐x_target
        StateVec x_offset_start = x0 - x_bar[0];
        StateVec x_offset_end   = x_target - x_bar[N_actual];
        for (int k = 0; k <= N_actual; ++k) {
            scalar_t alpha = static_cast<scalar_t>(k) / N_actual;
            StateVec x_offset = x_offset_start * (1.0 - alpha) + x_offset_end * alpha;
            x_bar[k] = x_bar[k] + x_offset;
        }
        init_ok = true;
    }

    if (!init_ok) {
        // 尝试G-FOLD获取初始控制轨迹
        float pos0_f[3]  = {(float)pos0[0], (float)pos0[1], (float)pos0[2]};
        float vel0_f[3]  = {(float)vel0[0], (float)vel0[1], (float)vel0[2]};
        float pos_t_f[3] = {(float)pos_target[0], (float)pos_target[1], (float)pos_target[2]};
        float vel_t_f[3] = {(float)vel_target[0], (float)vel_target[1], (float)vel_target[2]};
        float F_aero_f[3] = {0.0f, 0.0f, 0.0f};
        if (F_aero_n) {
            F_aero_f[0] = (float)F_aero_n[0];
            F_aero_f[1] = (float)F_aero_n[1];
            F_aero_f[2] = (float)F_aero_n[2];
        }

        bool gfold_ok = gfold.solve(pos0_f, vel0_f, pos_t_f, vel_t_f,
                                    static_cast<float>(m_current),
                                    static_cast<float>(tgo),
                                    F_aero_f, n_engines);

        if (gfold_ok) {
            // 用G-FOLD控制轨迹仿真出状态轨迹
            for (int k = 0; k < N_actual; ++k) {
                u_bar[k][0] = gfold.T_traj[k][0];
                u_bar[k][1] = gfold.T_traj[k][1];
                u_bar[k][2] = gfold.T_traj[k][2];
            }
            simulate_trajectory(x0, u_bar, wind, m_current, dt, N_actual, x_bar);
            init_ok = true;
        }
    }

    if (!init_ok) {
        // G-FOLD也失败: 直线插值
        for (int k = 0; k <= N_actual; ++k) {
            scalar_t alpha = static_cast<scalar_t>(k) / N_actual;
            x_bar[k] = x0 * (1.0 - alpha) + x_target * alpha;
        }
        // 默认控制: 悬停推力
        for (int k = 0; k < N_actual; ++k) {
            u_bar[k][0] = 0.0;
            u_bar[k][1] = 0.0;
            u_bar[k][2] = -m_current * scvx_atmos::G0;
        }
    }

    // === 障碍物绕行: 修改初始参考轨迹 ===
    if (multi_constraint) {
        apply_obstacle_detour(x_bar, N_actual);
    }

    // === prox-linear迭代 ===
    scalar_t total_solve_time = 0.0;
    bool converged = false;
    SCvxResult last_sub_result;
    last_sub_result.reset();
    int iteration = 0;

    for (iteration = 0; iteration < config.MAX_ITER; ++iteration) {
        // 熔断: 总超时检查
        if (total_solve_time > config.TOTAL_TIMEOUT) {
            break;
        }

        // 1. 线性化动力学
        linearize_dynamics(x_bar, u_bar, wind, m_current, dt, N_actual,
                          A_eff_buf, B_d_buf, const_eff_buf);

        // 2. 求解SOCP子问题
        auto t0 = std::chrono::steady_clock::now();
        SCvxResult sub_result = solve_scvx_subproblem(
            x0, x_target, A_eff_buf, B_d_buf, const_eff_buf,
            x_bar, u_bar, m_current, dt, T_max, T_min,
            N_actual, multi_constraint);
        auto t1 = std::chrono::steady_clock::now();
        scalar_t solve_time = std::chrono::duration<scalar_t>(t1 - t0).count();

        total_solve_time += solve_time;
        result.solve_time = total_solve_time;
        result.iterations = iteration + 1;

        // 熔断: 单步SOCP超时
        if (solve_time > config.SOLVE_TIMEOUT) {
            result.set_fail_reason("solve_timeout");
            break;
        }

        // 熔断: 求解失败
        if (!sub_result.success) {
            result.set_fail_reason("socp_failed");
            break;
        }

        // 3. 收敛判定 (每步平均归一化状态差)
        const scalar_t s_p = 100.0;
        const scalar_t s_v = 10.0;
        scalar_t x_diff_sum = 0.0;
        for (int k = 0; k <= N_actual; ++k) {
            scalar_t diff_pos = 0.0, diff_vel = 0.0;
            for (int i = 0; i < 3; ++i) {
                scalar_t dp = sub_result.x_traj[k][i] - x_bar[k][i];
                diff_pos += dp * dp;
                scalar_t dv = sub_result.x_traj[k][i + 3] - x_bar[k][i + 3];
                diff_vel += dv * dv;
            }
            x_diff_sum += std::sqrt(diff_pos) / s_p + std::sqrt(diff_vel) / s_v;
        }
        scalar_t x_diff_norm = x_diff_sum / (N_actual + 1);

        // 4. 更新参考轨迹
        for (int k = 0; k <= N_actual; ++k) {
            x_bar[k] = sub_result.x_traj[k];
        }
        for (int k = 0; k < N_actual; ++k) {
            u_bar[k] = sub_result.u_traj[k];
        }
        last_sub_result = sub_result;

        // 5. 收敛判定
        if (x_diff_norm < config.CONV_TOL) {
            converged = true;
            break;
        }
    }

    // === 熔断判定: 不收敛 → 降级 ===
    if (!converged) {
        result.fallback = true;
        fallback_count++;

        if (multi_constraint && has_warm_start) {
            // 多约束模式: 用上一次SCvx成功的控制轨迹
            for (int k = 0; k < N_actual; ++k) {
                result.u_traj[k] = last_u_bar[k];
            }
            result.success   = true;
            result.fallback  = true;
            result.converged = false;
            result.iterations = iteration;
            result.solve_time = total_solve_time;
            result.status    = SolveStatus::OPTIMAL;
            result.set_solver("last_scvx");
            result.set_fail_reason("not_converged");
            return result;
        } else {
            // 非多约束模式: 降级到G-FOLD
            float pos0_f[3]  = {(float)pos0[0], (float)pos0[1], (float)pos0[2]};
            float vel0_f[3]  = {(float)vel0[0], (float)vel0[1], (float)vel0[2]};
            float pos_t_f[3] = {(float)pos_target[0], (float)pos_target[1], (float)pos_target[2]};
            float vel_t_f[3] = {(float)vel_target[0], (float)vel_target[1], (float)vel_target[2]};
            float F_aero_f[3] = {0.0f, 0.0f, 0.0f};
            if (F_aero_n) {
                F_aero_f[0] = (float)F_aero_n[0];
                F_aero_f[1] = (float)F_aero_n[1];
                F_aero_f[2] = (float)F_aero_n[2];
            }

            bool gfold_ok = gfold.solve(pos0_f, vel0_f, pos_t_f, vel_t_f,
                                        static_cast<float>(m_current),
                                        static_cast<float>(tgo),
                                        F_aero_f, n_engines);

            if (gfold_ok) {
                for (int k = 0; k < N_actual; ++k) {
                    result.u_traj[k][0] = gfold.T_traj[k][0];
                    result.u_traj[k][1] = gfold.T_traj[k][1];
                    result.u_traj[k][2] = gfold.T_traj[k][2];
                    result.sigma_traj[k] = gfold.sigma_traj[k];
                }
                result.success   = true;
                result.fallback  = true;
                result.converged = false;
                result.iterations = iteration;
                result.solve_time = total_solve_time;
                result.cost      = gfold.last_cost;
                result.status    = SolveStatus::OPTIMAL;
                result.set_solver("GFOLD");
                result.set_fail_reason("not_converged");
                return result;
            } else {
                // 全部失败
                result.success   = false;
                result.fallback  = true;
                result.converged = false;
                result.iterations = iteration;
                result.solve_time = total_solve_time;
                result.status    = SolveStatus::OTHER_FAILURE;
                result.set_solver("none");
                result.set_fail_reason("all_failed");
                return result;
            }
        }
    }

    // === SCvx成功 ===
    solve_count++;
    result = last_sub_result;
    result.success   = true;
    result.fallback  = false;
    result.converged = true;
    result.iterations = iteration;
    result.solve_time = total_solve_time;
    result.N_solved  = N_actual;

    // 保存warm-starting
    for (int k = 0; k <= N_actual; ++k) {
        last_x_bar[k] = x_bar[k];
    }
    for (int k = 0; k < N_actual; ++k) {
        last_u_bar[k] = u_bar[k];
    }
    has_warm_start = true;

    return result;
}

#ifdef SCVX_USE_OSQP
// -----------------------------------------------------------------------
// OSQP求解接口 (当SCVX_USE_OSQP定义时启用)
//
// TODO: 实现完整的OSQP接口
//   1. 将SOCP问题转换为OSQP的QP形式 (通过松弛SOC约束)
//   2. 构造稀疏矩阵P, q, A, l, u
//   3. 调用OSQP求解
//   4. 将解转换为状态/控制轨迹
//
// 当前为框架代码, 实际使用时需要补充OSQP集成
// -----------------------------------------------------------------------
SCvxResult SCvxSolver::solve_with_osqp(
    const StateVec& x0, const StateVec& x_target,
    const std::array<StateMat, SCVX_N>& A_eff_list,
    const InputMat& B_d,
    const std::array<StateVec, SCVX_N>& const_eff_list,
    const std::array<StateVec, SCVX_N + 1>& x_bar_in,
    const std::array<ControlVec, SCVX_N>& u_bar_in,
    scalar_t m, scalar_t dt,
    scalar_t T_max, scalar_t T_min,
    int N_actual, bool multi_constraint) {

    // TODO: 实现OSQP接口
    // 当前降级到解析求解
    return solve_analytical(x0, x_target, A_eff_list, B_d, const_eff_list,
                            x_bar_in, u_bar_in, m, dt, T_max, T_min,
                            N_actual, multi_constraint);
}
#endif

}  // namespace falcon9

#endif  // FALCON9_SCVX_SOLVER_HPP
