"""
着陆制导模块 (G-FOLD SOCP联合优化 + slerp插值 + 死区保底 + 高空减速).
飞行剖面:
  Phase 1 DESCENT (h>1500m): 速度PD跟踪, 目标30m/s交给G-FOLD段
  Phase 2 G-FOLD (1500m→10m): SOCP联合优化推力大小+方向, slerp插值q_des, 线性插值油门
  Phase 3 DEADBAND (h<10m): 死点控制, 目标2m/s, RCS锁定垂直

核心架构(1.1 Step A: 油门回归SOCP):
  - 姿态控制器每0.01s主循环都执行, 绝不包在重求解if块里
  - Step A: 油门从SOCP解||T_vec[0]||/T_max提取, 删除匀减速解析公式.
    SOCP联合优化推力大小和方向, 代价函数加lambda_mag惩罚推力幅值变化率.
    油门也做1s线性插值(与q_des slerp同步), 防止重规划时跳变.
  - 修复23: G-FOLD每1s(100步)重求解(MPC), slerp插值q_des, Δθ方向平滑.
  - TVC和栅格舵指令每步都更新
  - 推力下限15%, THETA_GLIDE=10°, 终端速度容差6m/s, 水平位置10m
离散化: A_d=I+A*dt (精确, A²=0), B_d含dt²/(2m)项.
T/m非线性: 固定m_ref, 燃料消耗后验修正.
"""

import numpy as np
import cvxpy as cp
from . import quaternion_utils as qu
from . import rocket_params as rp

# G-FOLD物理参数
# Step B: T_MIN_FRAC从0.15改为0.40, 还原Merlin 1D真实节流下限.
#   T_min=0.40*845kN=338kN > 典型着陆重力275kN → 火箭无法悬停, 必须hover-slam.
#   SOCP必须靠倾斜飞行消耗多余推力.
# THETA_GLIDE: 20°不够! cos(20°)*338=317kN>275kN(重力), 火箭无法从近悬停状态下降.
#   物理要求: T_min*cos(θ) < mg → cos(θ) < 275/338=0.814 → θ > 35°.
#   取40°给余量: cos(40°)*338=259kN < 275kN, 火箭可下降(净下力16kN, a_down=0.57m/s²).
#   注: Step C加入气动力后可减小此角度, 气动阻力辅助减速.
T_MIN_FRAC = 0.40
THETA_GLIDE = np.radians(40.0)  # Step B: 40°, 物理要求>35°才能下降
# E5: 3发模式推力下限1014kN >> 重力363kN, 需要更大倾角才能下降.
#   cos(θ)*1014 < 363 → cos(θ) < 0.358 → θ > 69°. 取70°给余量.
#   但70°倾角过大(姿态控制困难), 实际3发模式用于高速制动(vz>40),
#   纯轴向推力即可制动, 不需大倾角. 仅在需要水平修正时用50°.
THETA_GLIDE_3 = np.radians(50.0)  # 3发模式指向锥 (允许更大倾角补偿高T_min)


# ---------- 工具: 推力方向 -> 期望姿态四元数 ----------
def q_des_from_thrust_dir(thrust_dir_n):
    """给定NED推力方向(单位向量) -> 期望姿态四元数(b->n).
    推力沿+Xb, 火箭垂直时Xb指向-Z_n(上). 故参考方向=up=[0,0,-1].
    q_des使 Xb在n系对齐thrust_dir_n. 用Q_VERT作为基准, 再叠加偏差旋转.
    修复22c: 旧版在世界系算旋转轴cross(up,d), 但Q_VERT⊗q_dev把轴当body系用,
      导致y方向倾角丢失(qd=0.5°实际应10°). 新版在body系算: 从[1,0,0]到C(Q_VERT)^T@d."""
    d = thrust_dir_n / np.linalg.norm(thrust_dir_n)
    # 期望body X方向 = C(Q_VERT)^T @ d (把世界系目标转到body系)
    C_qvert = qu.quat_to_rotmat(qu.Q_VERT)
    d_body = C_qvert.T @ d
    x_ref = np.array([1.0, 0.0, 0.0])
    cos_a = np.clip(np.dot(x_ref, d_body), -1.0, 1.0)
    axis = np.cross(x_ref, d_body)
    axis_n = np.linalg.norm(axis)
    if cos_a > 1.0 - 1e-9:
        return qu.Q_VERT.copy()
    if cos_a < -1.0 + 1e-9:
        return qu.quat_normalize(np.array([0.0, 1.0, 0.0, 0.0]))
    axis = axis / axis_n
    angle = np.arccos(cos_a)
    q_dev = np.array([np.cos(angle / 2),
                      axis[0] * np.sin(angle / 2),
                      axis[1] * np.sin(angle / 2),
                      axis[2] * np.sin(angle / 2)])
    return qu.quat_multiply(qu.Q_VERT, q_dev)


def quat_slerp(q0, q1, t):
    """球面线性插值. q0,q1为单位四元数, t∈[0,1]."""
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # 线性插值
        q = q0 + t * (q1 - q0)
        return qu.quat_normalize(q)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return qu.quat_normalize(s0 * q0 + s1 * q1)


# ---------- G-FOLD SOCP 规划器 ----------
class G_FOLD_Planner:
    """G-FOLD着陆轨迹优化 (SOCP, 6D状态+固定质量).
    状态: x=[pos_n(3), vel_n(3)], NED.
    控制: u=T_n(3), NED推力矢量.
    代价: sum(sigma) + 终端惩罚 + lambda_smooth*Δu² + lambda_mag*Δσ².
    约束: 推力上下限(SOC), 指向锥(SOC), 终端位置软约束, 速度软约束.
    1.1 Step A: 油门回归SOCP, lambda_mag惩罚推力幅值变化率, 油门从sigma提取.
    """
    def __init__(self, N=50, theta_max_rad=THETA_GLIDE, lambda_mag=0.0):
        self.N = N
        self.theta_max = theta_max_rad
        self.g = np.array([0.0, 0.0, rp.G0])  # NED重力+Z
        self.lambda_mag = lambda_mag  # Step B: 0.0(无损凸化), 移除sigma平滑
        self._debug = False  # Step B调试: MC测试前关闭
        # E6.3: 求解器硬化状态跟踪
        self._last_solve_info = {'status': 'optimal', 'success': True, 'solver': None,
                                 'solve_time': 0.0, 'cost': 0.0}
        self._solve_count = 0
        self._fail_count = 0
        self._solver_history = []  # [(solver_name, status, solve_time), ...]

    def solve(self, pos0, vel0, pos_target, vel_target, m_current, tgo,
              F_aero_n=None, n_engines=1):
        """求解G-FOLD. 返回 (T_traj[N,3], success_flag, info_dict).
        1.1 Step A修复: dt对齐重求解周期(1.0s). 旧版dt=tgo/N=0.25s@tgo=5s,
          但u0执行1.0s(100步), 是dt的4倍. u0为0.25s设计→悬停, 执行1.0s→1s悬停.
          新版: dt=max(1.0, tgo/N_max), N=tgo/dt. u0为1.0s设计, 执行1.0s, 匹配.
          N下限1: tgo短时(如0.57s@h=11m), N=1, u0为0.57s设计, 激进下降.
        1.1 Step C: 气动力作为已知常数加入动力学约束(零阶保持).
          F_aero_n: NED系气动力(N), 整个预测窗口内不变.
          动力学: v[k+1]=v[k]+(T[k]/m+g+F_aero/m)*dt. 不改变凸性.
        E5: n_engines参数支持多发动机配置.
          n_engines=1: T_min=338kN, T_max=845kN (精确着陆)
          n_engines=3: T_min=1014kN, T_max=2535kN (暴力制动)
          推力上下限按发动机数量缩放, 指向锥角度自适应."""
        # dt对齐重求解周期1.0s, N上限50防计算过慢, 下限1防过度短视
        # Step B修复: N从20增到50. 旧版N=20时tgo=46.5→dt=2.325s, 但执行1.0s,
        #   dt不匹配→火箭只执行43%规划冲量→下降时间翻倍(117s vs 预期50s).
        #   N=50时dt=max(1.0,46.5/50)=1.0s, 匹配执行周期, 无冲量损失.
        dt = max(1.0, tgo / self.N)
        N = max(1, min(self.N, int(round(tgo / dt))))
        dt = tgo / N  # 精确dt使N*dt=tgo
        m = m_current

        # 归一化
        s_p = 100.0
        s_v = 10.0
        s_T = 1e5

        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        # 精确离散化 (A²=0, A_d精确; B_d含dt²/2项)
        A_d = np.block([[I3, dt * (s_v / s_p) * I3],
                        [Z3, I3]])
        B_d = np.block([[0.5 * dt**2 * s_T / (m * s_p) * I3],
                        [dt * s_T / (m * s_v) * I3]])
        # 重力常数项 (NED系, Z向下为正, g=[0,0,9.81])
        g_d = np.array([0.0, 0.0, 0.5 * self.g[2] * dt**2 / s_p,
                        0.0, 0.0, self.g[2] * dt / s_v])

        # Step C: 气动力常数项 (零阶保持, 整个预测窗口不变)
        # F_aero_n: NED系气动力(N). 下降时阻力向上→Z分量为负.
        # 加速度 a_aero = F_aero_n / m, 加入g_d形成总常数项.
        # 验证: 火箭垂直下降vz>0, 阻力向上, F_aero_n[2] < 0 (NED Z向下为正).
        if F_aero_n is not None and np.linalg.norm(F_aero_n) > 0.1:
            a_aero = F_aero_n / m  # 气动力加速度 m/s²
            aero_d = np.array([
                0.5 * a_aero[0] * dt**2 / s_p,
                0.5 * a_aero[1] * dt**2 / s_p,
                0.5 * a_aero[2] * dt**2 / s_p,
                a_aero[0] * dt / s_v,
                a_aero[1] * dt / s_v,
                a_aero[2] * dt / s_v
            ])
            const_d = g_d + aero_d  # 重力+气动力总常数项
        else:
            const_d = g_d

        x0 = np.concatenate([pos0 / s_p, vel0 / s_v])
        x_target = np.concatenate([pos_target / s_p, vel_target / s_v])

        # 推力限幅 (归一化).
        # Step B: 还原T_min=40%约束. 理论方案1.1明确要求T_min=0.40*T_max.
        #   T_min=338kN > 重力275kN → 火箭无法悬停, 必须靠倾斜飞行消耗多余推力.
        #   THETA_GLIDE=40°(物理要求>35°): cos(40°)*338=259kN < 275kN, 可下降.
        # 松弛间隙(relaxation gap): SOC约束||u||<=sigma允许||u||<sigma.
        #   SOCP设sigma=T_min但用||u||<T_min(省油). 执行时若用||u||→物理不可行(发动机
        #   无法节流到T_min以下). 解决: 执行时用sigma(>=T_min)作为推力幅值, 不用||u||.
        #   SOCP用||u||规划轨迹, 执行用sigma → 推力比计划大 → 火箭上升 → MPC重解.
        #   Step D(无损凸化)将严格消除松弛间隙.
        # Step B关键修复: 用thrust_at_alt(h)而非THRUST_VAC!
        #   旧版T_max=THRUST_VAC=1200kN, T_min=0.40*1200=480kN.
        #   但着陆段(h<2000m)实际T_max=thrust_at_alt(h)≈845-855kN, T_min≈338-342kN.
        #   旧版T_min=480kN → 需tilt=55°(cosθ=275/480) > THETA_GLIDE=40° → 无法倾斜!
        #   SOCP被迫用松弛间隙(||u||<T_min)代替倾斜 → 执行clip到T_min → 火箭上升.
        #   修复: T_max=thrust_at_alt(h), T_min=0.40*T_max → tilt=36.5° < 40° → 可倾斜!
        h_current = -pos0[2]
        # E5: 多发动机推力上下限缩放.
        #   n_engines=1: T_min=338kN, T_max=845kN (1发精确着陆)
        #   n_engines=3: T_min=1014kN, T_max=2535kN (3发暴力制动)
        #   单台T_max=thrust_at_alt(h), T_min=T_MIN_FRAC*T_max.
        #   多台: 总T_max=n*T_max_single, 总T_min=n*T_min_single.
        T_max_single = rp.thrust_at_alt(h_current)
        T_max_phys = n_engines * T_max_single
        T_min_norm = n_engines * T_MIN_FRAC * T_max_single / s_T
        T_max_norm = T_max_phys / s_T
        # E5: 指向锥角度随发动机数自适应.
        #   1发: 40° (T_min*cos40°=259kN < 重力363kN, 可下降)
        #   3发: 50° (T_min*cos50°=652kN > 重力, 但3发用于高速制动, 倾角需求小)
        #   3发模式主要靠轴向推力制动, 大倾角仅用于水平修正.
        theta_max_eff = THETA_GLIDE_3 if n_engines >= 3 else self.theta_max

        # 优化变量
        x = cp.Variable((N + 1, 6))
        u = cp.Variable((N, 3))
        sigma = cp.Variable(N)

        constraints = [x[0] == x0]

        for k in range(N):
            constraints.append(x[k + 1] == A_d @ x[k] + B_d @ u[k] + const_d)
            constraints.append(cp.SOC(sigma[k], u[k]))
            constraints.append(sigma[k] >= T_min_norm)
            constraints.append(sigma[k] <= T_max_norm)
            # 指向约束: 限制推力倾角 <= theta_max.
            # Step B关键修复: 用||u_tangential|| <= tan(theta)*(-u_z), 不用sigma!
            #   旧版u_z + sigma*cos(theta) <= 0: 当sigma > ||u||(松弛间隙)时,
            #   约束变紧 → u_z <= -sigma*cos(theta) → 强制tilt≈0° → 无法倾斜下降.
            #   新版直接用u的切向/轴向分量, 与sigma无关, 正确限制倾角.
            #   SOC形式: ||u_tangential|| <= tan(theta) * (-u_z), 要求u_z<0(推力向上).
            constraints.append(cp.SOC(np.tan(theta_max_eff) * (-u[k, 2]), u[k, 0:2]))
            # Step B: 防止火箭上抛. T_min>重力时SOCP可能规划"先上升再下降"轨迹,
            #   MPC只执行首步(上升)→反复上升→火箭飞走. 约束vz≥-2m/s(归一化-0.2)
            #   允许小幅度上升(姿态调整)但禁止大角度上抛.
            constraints.append(x[k, 5] >= -0.2)  # vz ≥ -2 m/s (归一化)

        # 终端约束: 位置软约束, 速度软约束
        # Step B: 理论方案1.1要求放宽到20m/10m/s. T_min=40%时SOCP可行域缩小,
        #   太紧的终端约束会导致infeasible. 20m/10m/s给求解器足够腾挪空间.
        #   惩罚保持50000, 确保SOCP仍优先驱动到终端状态.
        eps_pos = cp.Variable(3)
        eps_vel = cp.Variable(3)
        constraints.append(eps_pos >= 0)
        constraints.append(eps_vel >= 0)
        # 终端位置在20m球内(归一化后0.2)
        constraints.append(cp.SOC(0.2 + cp.sum(eps_pos), x[N, 0:3] - x_target[0:3]))
        # 终端速度在10m/s球内(归一化后1.0)
        constraints.append(cp.SOC(1.0 + cp.sum(eps_vel), x[N, 3:6] - x_target[3:6]))

        # 代价: 1.1 Step A: 终端惩罚50000(从2000提高), 强制SOCP驱动到终端状态.
        #   旧版2000太低, 燃料成本(~100-200)与之相当, SOCP选择省油悬停.
        #   新版50000 >> 燃料成本, SOCP必须优先满足终端约束.
        # Step D(无损凸化验证): 纯sum(sigma) + 方向平滑 + 终端惩罚.
        #   理论: 代价只含sum(sigma)时, 最优解满足sigma=max(||u||,T_min).
        #   - ||u|| >= T_min时 → sigma=||u|| (无间隙, sigma/||u||=1, 无损凸化成立)
        #   - ||u|| < T_min时 → sigma=T_min > ||u|| (有间隙, 物理矛盾)
        #   去掉0.1*sum(sigma²): 二次项破坏无损凸化性质, 使sigma偏离||u||.
        #   保留0.1*方向平滑: 防数值抖动, 不影响无损凸化(平滑的是u不是sigma).
        #   去掉lambda_mag: sigma平滑项使sigma恒定, 破坏无损凸化.
        cost = cp.sum(sigma) + 50000.0 * (cp.sum(eps_pos) + cp.sum(eps_vel))
        lambda_smooth = 0.1  # Step D: 保留方向平滑防数值抖动
        for k in range(N - 1):
            cost += lambda_smooth * cp.sum_squares(u[k + 1] - u[k])

        prob = cp.Problem(cp.Minimize(cost), constraints)

        # E6.3: 求解器硬化 — CLARABEL优先 + SCS兜底 + 状态跟踪.
        #   工程决策: 不盲迁OSQP (OSQP不支持SOC约束, 转换复杂易错).
        #   CLARABEL (内点法): 数值稳定性好, 对条件数敏感的问题更可靠.
        #   SCS (一阶法): 对不可行问题更鲁棒, 不会崩溃.
        #   旧版SCS优先: 3发模式T_min/T_max比大时触发"inaccurate"警告.
        #   新版CLARABEL优先: 内点法精度更高, 避免警告.
        import time as _time
        solve_status = 'solver_crash'
        solver_used = None
        solve_t0 = _time.perf_counter()

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            solve_status = prob.status
            solver_used = prob.solver_stats.solver_name if prob.solver_stats else 'CLARABEL'
        except cp.error.SolverError:
            # CLARABEL失败 → SCS兜底
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=5000, eps=1e-4)
                solve_status = prob.status
                solver_used = prob.solver_stats.solver_name if prob.solver_stats else 'SCS'
            except Exception:
                solve_status = 'solver_crash'
                solver_used = 'none'
        except Exception:
            solve_status = 'solver_crash'
            solver_used = 'none'

        solve_time = _time.perf_counter() - solve_t0
        self._solve_count += 1

        # 更新求解器状态跟踪 (供SafetyMonitor查询)
        success = solve_status in ['optimal', 'optimal_inaccurate']
        if not success:
            self._fail_count += 1
        self._last_solve_info = {
            'status': solve_status, 'success': success,
            'solver': solver_used, 'solve_time': solve_time,
            'cost': prob.value if success else 0.0,
            'tgo': tgo, 'n_engines': n_engines,
        }
        # 保留最近50次求解历史 (防内存增长)
        self._solver_history.append((solver_used, solve_status, solve_time))
        if len(self._solver_history) > 50:
            self._solver_history.pop(0)

        if success:
            u_opt = (u.value * s_T).copy()
            sigma_opt = (sigma.value * s_T).copy()  # Step B: 返回sigma用于执行
            # Step B调试: 打印前5步sigma/||u||/gap
            # Step D验证: 打印σ[0]和||u[0]||间隙 (验证无损凸化)
            if hasattr(self, '_debug') and self._debug:
                print(f"    [SOLVE] status={prob.status} solver={solver_used} "
                      f"lambda_mag={self.lambda_mag} cost={prob.value:.4f} "
                      f"time={solve_time*1000:.1f}ms")
                for k in range(min(N, 5)):
                    un = np.linalg.norm(u_opt[k])
                    print(f"      k={k} ||u||={un/T_max_norm/s_T:.4f} "
                          f"sigma={sigma_opt[k]/s_T:.4f} "
                          f"T_min={T_min_norm:.4f} gap={sigma_opt[k]/s_T-max(un/s_T,T_min_norm):.4f}")

            # Step D: 无损凸化间隙验证打印 (已验证, 默认关闭)
            u0_norm = np.linalg.norm(u_opt[0])
            sigma0 = sigma_opt[0]
            gap = sigma0 / max(u0_norm, 1.0)  # sigma/||u||, =1无损, >1有间隙
            if hasattr(self, '_step_d_debug') and self._step_d_debug:
                print(f"  [STEP_D] sigma0={sigma0/s_T:.1f}N ||u0||={u0_norm/s_T:.1f}N "
                      f"T_min={T_min_norm*s_T:.1f}N T_max={T_max_norm*s_T:.1f}N "
                      f"gap={gap:.3f} {'无损✓' if gap < 1.05 else '有间隙✗'}")
            return u_opt, True, {'status': solve_status, 'cost': prob.value,
                                 'sigma_traj': sigma_opt, 'solver': solver_used,
                                 'solve_time': solve_time}
        else:
            return None, False, {'status': solve_status, 'tgo': tgo,
                                 'solver': solver_used, 'solve_time': solve_time}


# ---------- 死区保底控制器 ----------
class DeadbandController:
    """死区控制 (Step B: hover-slam bang-bang, 适配T_min=40%).
    NED: Z向下为正, 下降vz>0.
    Step B物理: T_min=40%→338kN > 重力275kN. 火箭无法悬停!
      - throttle=T_min时: 净上推力63kN, 持续减速(a_up≈2.25m/s²)
      - throttle=0时: 重力加速下降(a_down=9.81m/s²)
    策略: bang-bang控制vz在[target-deadband, target+deadband]内.
      vz > target+deadband(太快): throttle=T_min, 减速
      vz < target-deadband(太慢): throttle=0, 重力加速
      中间: 保持上一步输出(滞回防抖)
    水平修正: 小倾角PD(最大8°), 对抗风扰."""

    def __init__(self, target_vz=2.0, deadband=0.5):
        self.target_vz = target_vz
        self.deadband = deadband
        self.last_throttle = T_MIN_FRAC  # 初始=T_min

    def update(self, state, m, dt):
        vz = state[5]  # NED +Z向下, 下降vz>0

        # Step B: 连续PD油门控制. T_min=40%无法悬停.
        #   throttle = clip(T_min + Kp*(vz-target), T_min, 1.0)
        #   vz < target-deadband: throttle=0 (关机自由落体, 省油)
        #   中间: 保持上一步(滞回, 防止高频切换)
        Kp_db = 0.12
        if vz > self.target_vz + self.deadband:
            self.last_throttle = np.clip(T_MIN_FRAC + Kp_db * (vz - self.target_vz), T_MIN_FRAC, 1.0)
        elif vz < self.target_vz - self.deadband:
            self.last_throttle = 0.0
        # else: 保持last_throttle不变(滞回)
        throttle = self.last_throttle

        # 水平位置PD修正(最大8°, 低空安全, 对抗风扰)
        px, py = state[0], state[1]
        vx, vy = state[3], state[4]
        a_h = -0.08 * np.array([px, py]) - 0.5 * np.array([vx, vy])
        a_mag = np.linalg.norm(a_h)
        max_a = rp.G0 * np.sin(np.radians(8.0))  # DEADBAND段最大8°
        if a_mag > max_a:
            a_h = a_h * (max_a / a_mag)
            a_mag = max_a
        if a_mag > 0.01:
            tilt = np.arcsin(a_mag / rp.G0)
            dir_h = a_h / a_mag
            up_n = np.array([0.0, 0.0, -1.0])
            thrust_dir = up_n * np.cos(tilt) + np.array([dir_h[0], dir_h[1], 0.0]) * np.sin(tilt)
            q_des = q_des_from_thrust_dir(thrust_dir)
        else:
            q_des = qu.Q_VERT.copy()

        omega_des = np.zeros(3)
        return throttle, q_des, omega_des


# ---------- 着陆制导状态机 ----------
class LandingGuidance:
    """四阶段着陆制导: DESCENT -> G-FOLD -> DEADBAND -> LANDED.
    核心架构: 姿态控制器每步都执行, G-FOLD每1s(100步)重求解,
    slerp插值q_des + 线性插值throttle, 防止重规划时控制跳变.
    1.1 Step A: 油门从SOCP解提取, 联合优化推力大小+方向.
    """

    def __init__(self, gfold_N=50, dt=0.01):
        self.gfold = G_FOLD_Planner(N=gfold_N, theta_max_rad=THETA_GLIDE)
        self.deadband = DeadbandController()
        self.dt = dt
        self.phase = 'DESCENT'

        # E5: 1-3-1发动机点火策略
        #   3发: tgo>15s 或 vz>40 (暴力制动)
        #   1发: tgo<=15s 且 vz<40 (精确着陆)
        #   切换滞后: 3→1在tgo<13s, 1→3在tgo>18s (防抖)
        from .octaweb import LandingProfile
        self.landing_profile = LandingProfile()
        self.n_engines_current = 1  # DESCENT段用1发, G-FOLD段由LandingProfile决策

        # G-FOLD MPC状态 (修复23: 重新启用求解器提供q_des)
        self.solve_counter = 0          # 步数计数
        self.solve_period = 100         # 100步=1s重求解 (核心铁律7)
        self.last_solve_time = -10.0
        self.last_T_traj = None
        self.last_solve_success = False
        self.solve_happened_this_step = False  # E6.3: 本步是否触发求解 (供SafetyMonitor)

        # q_des平滑状态 (修复23: slerp插值, 防止求解器间控制跳跃)
        self.q_des_prev = qu.Q_VERT.copy()
        self.q_des_current = qu.Q_VERT.copy()
        self.q_des_target = qu.Q_VERT.copy()

        # throttle平滑状态 (1.1 Step A: 线性插值, 与q_des slerp同步)
        self.throttle_prev = 0.5
        self.throttle_current = 0.5
        self.throttle_target = 0.5

        # Step B: bang-bang执行状态 (当SOCP返回||u||<T_min时)
        #   T_min=40%>重力, 发动机无法节流到T_min以下. SOCP用松弛间隙规划||u||<T_min,
        #   执行时不能用||u||(物理不可行), 也不能clip到T_min(推力过大→上升).
        #   正确做法: bang-bang(0或T_min), 占空比平均=||u||, 匹配SOCP规划轨迹.
        #   用vz反馈闭环: vz>3.5→T_min减速, vz<2.5→0自由落体, 中间滞回.
        self.bang_mode = False
        self.bang_last_throttle = T_MIN_FRAC

        # G-FOLD进入状态
        self.gfold_entry_pos = None
        self.gfold_entry_vel = None
        self.gfold_t0 = 0.0
        self.gfold_tgo_fixed = 20.0
        self.landed = False
        # Step C: 存储上一步气动力(NED系), 零阶保持传入SOCP
        self.F_aero_n_prev = np.zeros(3)

    def update(self, state, fuel_mass, t, dt):
        """主更新: 返回 (throttle, q_des, omega_des, tvc_gimbal_cmd, phase).
        姿态控制器由调用方每步执行, 这里只返回q_des和throttle和tvc_gimbal.
        """
        pos_n = state[0:3]
        vel_n = state[3:6]
        q = state[6:10]
        h = -pos_n[2]
        m, cg_x, I_body = rp.mass_properties(fuel_mass)
        vz = vel_n[2]  # NED +Z向下, 下降vz>0

        tvc_gimbal = np.zeros(2)

        # === 阶段转换 ===
        # 修复11: G-FOLD入口从1000m提高到1500m, 给水平修正更多时间
        if self.phase == 'DESCENT' and h <= 1500.0:
            self.phase = 'G-FOLD'
            self.gfold_entry_pos = pos_n.copy()
            self.gfold_entry_vel = vel_n.copy()
            self.last_solve_time = -10.0
            self.gfold_t0 = t
            self.solve_counter = 0
            # Step B: G-FOLD终端从h=10m降到h=5m, 减少DEADBAND段长度.
            #   T_min=40%时DEADBAND无法悬停, 缩短DEADBAND时间降低风险.
            # tgo公式: 匀减速时间 tgo=2d/(v0+vt), d=h-h_terminal, vt=3m/s.
            # Step A修复: vz下限0.5, 避免悬停时tgo过短.
            vz_entry = max(vel_n[2], 0.5)
            vz_terminal = 3.0
            tgo_kin = 2.0 * (h - 5.0) / (vz_entry + vz_terminal)
            self.gfold_tgo_fixed = np.clip(tgo_kin, 10.0, 55.0)

        # Step B: DEADBAND入口从10m降到5m, 缩短无法悬停的DEADBAND段.
        if self.phase == 'G-FOLD' and h <= 5.0:
            self.phase = 'DEADBAND'

        if self.phase == 'DEADBAND' and h <= 0.5:
            self.phase = 'LANDED'
            self.landed = True

        # === 各阶段逻辑 ===
        if self.phase == 'DESCENT':
            throttle, q_des = self._descent_control(h, vz, m, t, pos_n, vel_n)
            omega_des = np.zeros(3)
            # E5: DESCENT段用1发 (高空巡航, 不需3发制动)
            self.n_engines_current = 1
            # 更新q_des平滑状态
            self.q_des_current = q_des.copy()
            self.q_des_prev = q_des.copy()
            self.q_des_target = q_des.copy()

        elif self.phase == 'G-FOLD':
            throttle, q_des, omega_des, tvc_gimbal = self._gfold_control(
                state, pos_n, vel_n, q, m, t, dt)

        elif self.phase == 'DEADBAND':
            throttle, q_des, omega_des = self.deadband.update(state, m, dt)
            self.q_des_current = q_des.copy()
            # E5: DEADBAND段用1发 (精确着陆)
            self.n_engines_current = 1

        else:  # LANDED
            throttle = 0.0
            q_des = qu.Q_VERT.copy()
            omega_des = np.zeros(3)
            self.n_engines_current = 0

        return throttle, q_des, omega_des, tvc_gimbal, self.phase

    def _descent_control(self, h, vz, m, t, pos_n, vel_n):
        """高空减速: 速度PD跟踪, 目标30m/s交给G-FOLD.
        修复13: 增加水平速度阻尼, 防止px在DESCENT段从50m增长到150m.
        NED: vz>0为下降. h高时目标80m/s, h接近1000m时目标30m/s.
        修复1: throttle改为[0,1]直接推力百分比."""
        # 目标速度: 线性插值, h=2000m→80m/s, h=1000m→30m/s
        if h > 2000.0:
            v_target = 80.0
        elif h > 1000.0:
            v_target = 30.0 + 50.0 * (h - 1000.0) / 1000.0  # 1000m→30, 2000m→80
        else:
            v_target = 30.0
        v_err = vz - v_target  # vz>v_target=太快, 需加油门减速
        # PD: a_needed = -Kp * v_err (v_err>0时a<0即向上减速)
        a_needed = -0.5 * v_err
        # 修复5: NED中 a_actual = G0 - T/m (重力下,推力上). 要a_actual=a_needed:
        #   T/m = G0 - a_needed  =>  T = m*(G0 - a_needed)
        # 旧版 T=m*(G0+a_needed) 符号错误, 导致太快时反而减推力!
        T_needed = m * (rp.G0 - a_needed)
        T_sl = rp.thrust_at_alt(h)
        throttle = np.clip(T_needed / T_sl, T_MIN_FRAC, 1.0)

        # 修复13: 水平速度阻尼. 小倾角(最大5°)对抗水平速度, 防止px增长.
        # DESCENT段不修正水平位置(交给G-FOLD), 但要防止水平速度持续推走箭体.
        v_horiz = np.array([vel_n[0], vel_n[1]])
        v_horiz_mag = np.linalg.norm(v_horiz)
        if v_horiz_mag > 1.0:
            # 倾斜角与水平速度成正比, 最大5°, 方向与速度相反
            tilt_deg = min(5.0, v_horiz_mag * 0.3)
            tilt_rad = np.radians(tilt_deg)
            horiz_dir = -v_horiz / v_horiz_mag  # 与速度相反方向
            up_n = np.array([0.0, 0.0, -1.0])
            thrust_dir = up_n + tilt_rad * np.array([horiz_dir[0], horiz_dir[1], 0.0])
            thrust_dir = thrust_dir / np.linalg.norm(thrust_dir)
            q_des = q_des_from_thrust_dir(thrust_dir)
        else:
            q_des = qu.Q_VERT.copy()
        return throttle, q_des

    def _gfold_control(self, state, pos_n, vel_n, q, m, t, dt):
        """G-FOLD段: SOCP联合优化推力大小+方向 + slerp/线性插值.
        1.1 Step A: 油门从SOCP解||T_vec[0]||/T_max提取, 删除匀减速解析公式.
          SOCP联合优化推力大小和方向, 代价函数加lambda_mag惩罚幅值变化率.
          油门也做1s线性插值(与q_des slerp同步), 防止重规划时跳变.
        修复23: 每1s(100步)重求解(MPC), slerp插值q_des, Δθ方向平滑.
        核心铁律7: G-FOLD每1秒重求解(MPC), 取首控制量执行.
        E5: 1-3-1发动机切换. 3发模式用于vz>40暴力制动, 1发用于精确着陆.
          3发模式终端约束放宽(vz=40), 因T_min=1014kN无法减速到3m/s.
          1发模式终端约束严格(vz=3, h=5), hover-slam精确着陆."""
        h = -pos_n[2]
        vz = vel_n[2]
        T_max_single = rp.thrust_at_alt(h)

        # === E5: 1-3-1发动机配置决策 ===
        # 计算当前tgo (用于LandingProfile决策)
        vz_for_tgo = max(vz, 0.5)
        tgo_est = 2.0 * (h - 5.0) / (vz_for_tgo + 3.0)
        tgo_est = np.clip(tgo_est, 1.0, 55.0)
        n_engines = self.landing_profile.decide_engine_config(h, vz, tgo_est, t)
        self.n_engines_current = n_engines
        T_max = n_engines * T_max_single  # 总推力上限

        # E6.3: 本步是否触发求解标志 (重置)
        self.solve_happened_this_step = False

        # === G-FOLD SOCP求解 (每100步=1s) ===
        if self.solve_counter % self.solve_period == 0:
            self.solve_happened_this_step = True  # E6.3: 标记本步有求解
            # E5: 终端目标随发动机配置变化.
            #   1发: 精确着陆, h=5m, vz=3m/s (hover-slam)
            #   3发: 暴力制动段, 不要求精确着陆. 目标vz=40(切换阈值),
            #        h=当前-100(给1发段留空间). 避免SOCP因T_min过大而不可行.
            if n_engines >= 3:
                pos_target = np.array([0.0, 0.0, -(h - 100.0)])  # h_target=h-100m
                vel_target = np.array([0.0, 0.0, 40.0])  # vz_target=40 (切换阈值)
            else:
                pos_target = np.array([0.0, 0.0, -5.0])
                vel_target = np.array([0.0, 0.0, 3.0])

            # 剩余时间: 用当前状态算匀减速tgo (Bug 6修复)
            # E5: 3发模式tgo更短(只规划到vz=40), 1发模式tgo到vz=3.
            # E5修复: 1发模式tgo上限25s(原55s), 强制更快制动.
            #   旧版55s → SOCP燃料最优解缓慢下降(117s着陆, 不真实).
            #   新版25s → SOCP必须更快制动, 着陆时间~30s(接近F9实际).
            vz_now = max(vel_n[2], 0.5)
            vz_terminal = vel_target[2]
            tgo_remaining = 2.0 * (h - (-pos_target[2])) / (vz_now + vz_terminal)
            tgo_max = 20.0 if n_engines == 1 else 55.0  # 1发限20s, 3发限55s
            tgo_remaining = np.clip(tgo_remaining, 1.0, tgo_max)

            T_traj, success, info = self.gfold.solve(
                pos_n, vel_n, pos_target, vel_target, m, tgo_remaining,
                F_aero_n=self.F_aero_n_prev, n_engines=n_engines)

            # Step C调试: 打印气动力方向 (已验证, 注释)
            # fa = self.F_aero_n_prev
            # if np.linalg.norm(fa) > 0.1 and self.solve_counter % 500 == 0:
            #     print(f"  [AERO] t={t:.1f} h={h:.1f} vz={vz:.2f} "
            #           f"F_aero_n=[{fa[0]:.0f},{fa[1]:.0f},{fa[2]:.0f}]N "
            #           f"|F|={np.linalg.norm(fa):.0f}N "
            #           f"vz_sign={'下降' if vz>0 else '上升'} "
            #           f"aero_z_sign={'向上(正确)' if fa[2]<0 else '向下(错误!)'}")

            # Step B调试: 打印SOCP求解状态 (MC测试前注释)
            # if success and T_traj is not None:
            #     u0 = T_traj[0]
            #     u_norm = np.linalg.norm(u0)
            #     sigma0 = info.get('sigma_traj', np.array([u_norm]))[0]
            #     tilt_u = np.degrees(np.arctan2(np.hypot(u0[0], u0[1]), -u0[2]))
            #     T_min_phys = T_MIN_FRAC * T_max
            #     bang_flag = "BANG" if u_norm < T_min_phys - 1.0 else "CONT"
            #     print(f"  [SOCP-OK] t={t:.1f} h={h:.1f} vz={vz:.2f} tgo={tgo_remaining:.1f} "
            #           f"||u||={u_norm/T_max:.3f} sigma={sigma0/T_max:.3f} tilt={tilt_u:.1f}° "
            #           f"{bang_flag} N={len(T_traj)}")
            # else:
            #     print(f"  [SOCP-FAIL] t={t:.1f} h={h:.1f} vz={vz:.2f} tgo={tgo_remaining:.1f} "
            #           f"status={info.get('status','?')}")

            if success and T_traj is not None:
                # 取首控制量
                u0 = T_traj[0]
                u_norm = np.linalg.norm(u0)
                T_min_phys = T_MIN_FRAC * T_max
                # Step B: bang-bang执行判断.
                #   ||u|| >= T_min: 连续油门, 从SOCP提取throttle=||u||/T_max
                #   ||u|| < T_min: SOCP用了松弛间隙(规划||u||<T_min但sigma=T_min).
                #     发动机无法节流到T_min以下, 执行不能用||u||(物理不可行).
                #     也不能clip到T_min(推力>计划→火箭上升→SOCP不可行→死锁).
                #     正确: bang-bang(0或T_min), 占空比d=||u||/T_min, 平均推力=||u||.
                #     用vz反馈闭环实现, 自然找到正确占空比, 无需开环计时.
                # E5: 1发模式(T_min<g)强制bang-bang. MPC燃料最优解会卡在低推力段
                #   (SOCP返回||u||≈T_min<g → 火箭加速下降 → tgo不收敛 → 117s着陆).
                #   bang-bang(0或T_max)是T_min<g<T_max的理论最优解, 收敛快.
                if n_engines == 1:
                    self.bang_mode = True
                    throttle_new = None
                elif u_norm >= T_min_phys - 1.0:  # -1N容差防数值抖动
                    self.bang_mode = False
                    throttle_new = np.clip(u_norm / T_max, T_MIN_FRAC, 1.0)
                else:
                    self.bang_mode = True
                    throttle_new = None  # bang-bang模式, 不用线性插值

                # 推力方向 -> q_des (无论是否bang-bang, 都用SOCP方向)
                if u_norm > 1.0:
                    thrust_dir = u0 / u_norm
                    q_des_new = q_des_from_thrust_dir(thrust_dir)
                else:
                    q_des_new = qu.Q_VERT.copy()

                # 更新slerp/线性插值状态: 从当前值平滑过渡到新值
                self.q_des_prev = self.q_des_current.copy()
                self.q_des_target = q_des_new
                if not self.bang_mode:
                    self.throttle_prev = self.throttle_current
                    self.throttle_target = throttle_new
                self.last_solve_time = t
                self.last_solve_success = True
                self.last_T_traj = T_traj
            else:
                # 求解失败: 保持上一周期目标, 插值继续完成过渡
                self.last_solve_success = False
                # print(f"  [SOCP-FAIL] t={t:.2f} h={h:.1f} vz={vz:.2f} tgo={tgo_remaining:.2f}")

        # === slerp插值q_des + 油门计算 ===
        solve_dt = t - self.last_solve_time
        alpha = min(1.0, max(0.0, solve_dt / 1.0))
        q_des = quat_slerp(self.q_des_prev, self.q_des_target, alpha)
        self.q_des_current = q_des.copy()

        if self.bang_mode:
            # Step B: 连续PD油门 (T_min=40%>重力, 无法悬停).
            #   改进: target_vz高度相关, 高空允许快降, 低空减速.
            #   - target_vz = max(3.0, h/30): h=90m→vz=3, h=300m→vz=10, h=900m→vz=30
            #   - vz > target: throttle = clip(T_min + Kp*(vz-target), T_min, 1.0)
            #   - vz < target-deadband: throttle=0 (关机自由落体, 省油)
            #   - 中间: 保持上一步(滞回防抖)
            #   修复: 旧版固定target_vz=3导致h=900m时vz=3, 下降300s超时.
            # E5: 1发模式target_vz=40(切换阈值), 1发模式target_vz=3(精确着陆).
            #   3发T_min=1014kN>>重力, bang-bang目标为制动到vz=40后切换1发.
            # E5修复: 1发模式用hover-slam剖面 target_vz=2*sqrt(h).
            #   旧版h/30在低空太保守(h=90→vz=3, 下降30s).
            #   2*sqrt(h): h=400→40, h=100→20, h=25→10, h=4→4 (恒定减速度2m/s²).
            #   着陆时间从98s降到~30s.
            if n_engines >= 3:
                target_vz_bb = max(40.0, h / 20.0)  # 3发: 目标vz=40
                Kp_bb = 0.05  # 3发增益小, T_min已很大
            else:
                target_vz_bb = max(3.0, 2.0 * np.sqrt(max(h, 0.0)))  # 1发: hover-slam剖面
                Kp_bb = 0.12
            deadband_bb = 0.5
            if vz > target_vz_bb + deadband_bb:
                throttle = np.clip(T_MIN_FRAC + Kp_bb * (vz - target_vz_bb), T_MIN_FRAC, 1.0)
                self.bang_last_throttle = throttle
            elif vz < target_vz_bb - deadband_bb:
                throttle = 0.0
                self.bang_last_throttle = 0.0
            else:
                throttle = self.bang_last_throttle
            self.throttle_current = throttle
        else:
            throttle = (1.0 - alpha) * self.throttle_prev + alpha * self.throttle_target
            self.throttle_current = throttle

        # 安全保护: 绝对物理阈值 (Step B: THETA_GLIDE=40°+2°余量=42°)
        # E5: 3发模式允许更大倾角(50°+2°余量=52°)
        tilt_limit = 52.0 if n_engines >= 3 else 42.0
        tilt_actual = qu.tilt_angle_from_vertical(state[6:10])
        if np.degrees(tilt_actual) > tilt_limit:
            q_des = qu.Q_VERT.copy()
            self.q_des_current = q_des.copy()

        omega_des = np.zeros(3)

        # TVC gimbal: 由姿态环计算(attitude_control.py), 制导环返回0
        tvc_gimbal = np.zeros(2)

        self.solve_counter += 1
        return throttle, q_des, omega_des, tvc_gimbal