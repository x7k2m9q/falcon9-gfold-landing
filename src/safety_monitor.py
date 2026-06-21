"""E6: 安全监控器 + 异常处理框架 (工程硬化).

理论方案2.0 E6节:
  6.2 SafetyMonitor: G-FOLD失效时的兜底决策
    - 可达集检查 (能量法: vz² ≤ 2*a_brake*h)
    - SOCP不可行检测 (跟踪last_solve_status)
    - 燃料告警 (FUEL_RESERVE阈值)
    - 姿态超限 (tilt > 15° → RCS全力回正)
  6.3 FlightController: 异常处理框架
    - 输入合理性检验 (NaN/Inf/范围)
    - EKF状态检验 (NaN/Inf/协方差正定)
    - 解的合理性检验 (推力范围/方向/幅值)
    - 分级兜底: SensorFault → StateEstimationFault → SolverFault → SafeMode

工程决策 (E6.3求解器硬化):
  不盲迁OSQP. 理由:
    1. OSQP原生不支持SOC约束, SOCP→QP转换复杂易错
    2. 当前SCS+CLARBEL组合5/5 MC成功, "inaccurate"警告已验证不影响结果
    3. CLARABEL(内点法)数值稳定性优于SCS(一阶法), 作为主求解器
    4. SCS作为兜底(对不可行问题更鲁棒)
    5. 风险收益比: 迁移风险 > 速度/稳定性收益
  改为: CLARABEL优先 + SCS兜底 + 求解器状态跟踪 + 失败计数熔断.

坐标系: NED (Z向下为正), 下降vz>0, 推力向上=u_z<0.
"""
import numpy as np
from . import rocket_params as rp
from . import quaternion_utils as qu


# ============================================================
# 异常类型层次
# ============================================================
class FlightException(Exception):
    """飞行异常基类."""
    pass


class SensorFault(FlightException):
    """传感器故障: NaN/Inf/超量程."""
    pass


class StateEstimationFault(FlightException):
    """状态估计故障: EKF发散/协方差爆炸."""
    pass


class SolverFault(FlightException):
    """求解器故障: SOCP不可行/数值崩溃."""
    pass


class SafetyViolation(FlightException):
    """安全约束违反: 不可达/姿态超限/燃料耗尽."""
    pass


# ============================================================
# SafetyMonitor
# ============================================================
class SafetyMonitor:
    """G-FOLD失效时的兜底决策器.

    检查项 (优先级从高到低):
      1. 姿态超限 (tilt > 15°): EMERGENCY_RECOVER — RCS全力回正
      2. 可达集违反 (vz² > 2*a_brake*h): ABORT_MAX_BRAKE — 全发满推力刹车
      3. SOCP不可行 (last_solve_status=infeasible): FALLBACK_PD — 退化PD控制
      4. 燃料告警 (fuel < FUEL_RESERVE): MIN_FUEL_GLIDE — 最省油滑翔
      5. 全部正常: NOMINAL

    返回状态字符串, FlightController根据状态选择控制策略.
    """

    # 安全阈值
    TILT_LIMIT_DEG = 15.0          # 姿态超限阈值 (理论方案2.0: 15°)
    TILT_EMERGENCY_DEG = 30.0      # 紧急姿态阈值 (RCS全力)
    FUEL_RESERVE = 500.0           # 燃料储备 kg (最低着陆储备)
    FUEL_CRITICAL = 100.0          # 临界燃料 kg (必须立即着陆)
    VZ_MAX_DESCENT = 120.0         # 最大下降速度 m/s (超此值不可恢复)
    H_MIN_BRAKE = 50.0             # 最低制动高度 m (低于此高度不允许max_brake)

    # 状态枚举
    NOMINAL = "NOMINAL"
    EMERGENCY_RECOVER = "EMERGENCY_RECOVER"
    ABORT_MAX_BRAKE = "ABORT_MAX_BRAKE"
    FALLBACK_PD = "FALLBACK_PD"
    MIN_FUEL_GLIDE = "MIN_FUEL_GLIDE"
    SAFE_MODE = "SAFE_MODE"

    def __init__(self):
        self.last_solve_status = "optimal"
        self.solve_fail_count = 0
        self.solve_fail_streak = 0  # 连续失败计数
        self.last_status = self.NOMINAL
        self.last_violation = ""
        # 熔断: 连续失败超过阈值后强制PD兜底
        self.fail_threshold_circuit = 3  # 连续3次失败熔断

    def update_solve_status(self, status, success):
        """由guidance调用, 更新SOCP求解状态."""
        self.last_solve_status = status
        if success:
            self.solve_fail_streak = 0
        else:
            self.solve_fail_count += 1
            self.solve_fail_streak += 1

    def check(self, state, fuel, tgo=10.0, n_engines=1):
        """执行所有安全检查, 返回 (status, violation_msg).

        参数:
          state: 13D状态向量 [pos_n(3), vel_n(3), q(4), omega_b(3)]
          fuel: 剩余燃料 kg
          tgo: 估计剩余时间 s (用于可达集)
          n_engines: 当前发动机数 (影响制动能力)
        """
        pos_n = state[0:3]
        vel_n = state[3:6]
        q = state[6:10]
        h = -pos_n[2]
        vz = vel_n[2]  # NED +Z向下, 下降vz>0
        v_horiz = np.hypot(vel_n[0], vel_n[1])

        # === 1. 姿态超限检查 (最高优先级, 立即危及箭体结构) ===
        tilt_rad = qu.tilt_angle_from_vertical(q)
        tilt_deg = np.degrees(tilt_rad)
        if tilt_deg > self.TILT_EMERGENCY_DEG:
            self.last_status = self.EMERGENCY_RECOVER
            self.last_violation = "tilt=%.1f° > %.1f° (紧急)" % (tilt_deg, self.TILT_EMERGENCY_DEG)
            return self.EMERGENCY_RECOVER, self.last_violation
        if tilt_deg > self.TILT_LIMIT_DEG:
            self.last_status = self.EMERGENCY_RECOVER
            self.last_violation = "tilt=%.1f° > %.1f°" % (tilt_deg, self.TILT_LIMIT_DEG)
            return self.EMERGENCY_RECOVER, self.last_violation

        # === 2. 可达集检查 (能量法) ===
        # 最大制动减速度: a_brake = T_max/m - g (向上为正, 减速下降)
        # 可达条件: vz² ≤ 2 * a_brake * h (动能 ≤ 势能差*制动能力)
        # 若vz² > 2*a_brake*h → 当前推力无法在h内制动到vz=0 → 必须max_brake
        m, _, _ = rp.mass_properties(fuel)
        T_max_single = rp.thrust_at_alt(max(h, 0.0))
        T_max = n_engines * T_max_single
        a_brake = T_max / m - rp.G0  # 净向上加速度
        if a_brake > 0.1 and h > self.H_MIN_BRAKE:
            vz_max_stoppable = np.sqrt(2.0 * a_brake * h)
            if vz > vz_max_stoppable * 1.1:  # 10%余量
                self.last_status = self.ABORT_MAX_BRAKE
                self.last_violation = "vz=%.1f > vz_max=%.1f (不可达)" % (vz, vz_max_stoppable)
                return self.ABORT_MAX_BRAKE, self.last_violation

        # 下降速度绝对上限 (超此值气动加热/结构载荷过大)
        if vz > self.VZ_MAX_DESCENT:
            self.last_status = self.ABORT_MAX_BRAKE
            self.last_violation = "vz=%.1f > %.1f (超极限)" % (vz, self.VZ_MAX_DESCENT)
            return self.ABORT_MAX_BRAKE, self.last_violation

        # === 3. SOCP不可行检测 (连续失败熔断) ===
        if self.solve_fail_streak >= self.fail_threshold_circuit:
            self.last_status = self.FALLBACK_PD
            self.last_violation = "SOCP连续失败%d次 (熔断)" % self.solve_fail_streak
            return self.FALLBACK_PD, self.last_violation

        if self.last_solve_status in ("infeasible", "infeasible_inaccurate"):
            self.last_status = self.FALLBACK_PD
            self.last_violation = "SOCP status=%s" % self.last_solve_status
            return self.FALLBACK_PD, self.last_violation

        # === 4. 燃料告警 ===
        if fuel < self.FUEL_CRITICAL:
            self.last_status = self.MIN_FUEL_GLIDE
            self.last_violation = "fuel=%.0fkg < %.0fkg (临界)" % (fuel, self.FUEL_CRITICAL)
            return self.MIN_FUEL_GLIDE, self.last_violation
        if fuel < self.FUEL_RESERVE:
            self.last_status = self.MIN_FUEL_GLIDE
            self.last_violation = "fuel=%.0fkg < %.0fkg" % (fuel, self.FUEL_RESERVE)
            return self.MIN_FUEL_GLIDE, self.last_violation

        # === 全部正常 ===
        self.last_status = self.NOMINAL
        self.last_violation = ""
        return self.NOMINAL, ""

    def get_fallback_throttle(self, status, h, vz, n_engines):
        """根据安全状态返回兜底油门指令.

        ABORT_MAX_BRAKE: 满推力 (1.0)
        EMERGENCY_RECOVER: 满推力 (1.0, 用推力矢量回正)
        MIN_FUEL_GLIDE: 最小推力 (T_MIN_FRAC, 省油滑翔)
        FALLBACK_PD: 由PD控制决定 (返回None, 由调用方PD计算)
        NOMINAL: None (正常制导)
        """
        if status == self.ABORT_MAX_BRAKE:
            return 1.0
        if status == self.EMERGENCY_RECOVER:
            return 1.0
        if status == self.MIN_FUEL_GLIDE:
            return rp.THRUST_MIN_PCT
        return None  # FALLBACK_PD 或 NOMINAL, 由调用方决定


# ============================================================
# 输入/状态/解的合理性检验
# ============================================================
class InputValidator:
    """输入合理性检验工具集."""

    # 传感器量程
    ACCEL_MAX = 100.0       # 加速度量程 100g (实际F9约5g)
    GYRO_MAX = 10.0         # 角速度量程 10 rad/s
    GPS_POS_MAX = 1e5       # 位置量程 100km
    GPS_VEL_MAX = 1e3       # 速度量程 1000 m/s
    RADAR_ALT_MAX = 1e4     # 高度量程 10km

    @classmethod
    def validate_sensor(cls, gyro_meas, accel_meas, gps_pos=None, gps_vel=None, radar_alt=None):
        """检验传感器读数合理性. 异常则抛SensorFault."""
        # NaN/Inf检查
        for name, val in [("gyro", gyro_meas), ("accel", accel_meas)]:
            if val is None:
                raise SensorFault("%s is None" % name)
            if not np.all(np.isfinite(val)):
                raise SensorFault("%s contains NaN/Inf: %s" % (name, val))

        # 量程检查
        if np.linalg.norm(accel_meas) > cls.ACCEL_MAX:
            raise SensorFault("accel超量程: |a|=%.2f > %.2f" %
                              (np.linalg.norm(accel_meas), cls.ACCEL_MAX))
        if np.linalg.norm(gyro_meas) > cls.GYRO_MAX:
            raise SensorFault("gyro超量程: |w|=%.2f > %.2f" %
                              (np.linalg.norm(gyro_meas), cls.GYRO_MAX))

        if gps_pos is not None:
            if not np.all(np.isfinite(gps_pos)):
                raise SensorFault("gps_pos NaN/Inf: %s" % gps_pos)
            if np.linalg.norm(gps_pos) > cls.GPS_POS_MAX:
                raise SensorFault("gps_pos超量程: %s" % gps_pos)
        if gps_vel is not None:
            if not np.all(np.isfinite(gps_vel)):
                raise SensorFault("gps_vel NaN/Inf: %s" % gps_vel)
            if np.linalg.norm(gps_vel) > cls.GPS_VEL_MAX:
                raise SensorFault("gps_vel超量程: %s" % gps_vel)
        if radar_alt is not None:
            if not np.isfinite(radar_alt):
                raise SensorFault("radar_alt NaN/Inf: %s" % radar_alt)
            if radar_alt < -10.0 or radar_alt > cls.RADAR_ALT_MAX:
                raise SensorFault("radar_alt超量程: %.2f" % radar_alt)
        return True

    @classmethod
    def validate_ekf_state(cls, p, v, q, bg=None, ba=None, P=None):
        """检验EKF估计状态合理性. 异常则抛StateEstimationFault."""
        for name, val in [("p", p), ("v", v), ("q", q)]:
            if val is None:
                raise StateEstimationFault("%s is None" % name)
            if not np.all(np.isfinite(val)):
                raise StateEstimationFault("%s NaN/Inf: %s" % (name, val))

        # 四元数范数
        q_norm = np.linalg.norm(q)
        if abs(q_norm - 1.0) > 0.1:
            raise StateEstimationFault("q范数异常: |q|=%.4f" % q_norm)

        # 位置/速度量程
        if np.linalg.norm(p) > cls.GPS_POS_MAX:
            raise StateEstimationFault("p超量程: %s" % p)
        if np.linalg.norm(v) > cls.GPS_VEL_MAX:
            raise StateEstimationFault("v超量程: %s" % v)

        # 协方差正定性检查 (对角线必须为正)
        if P is not None:
            diag = np.diag(P)
            if np.any(diag < 0):
                raise StateEstimationFault("P对角线负: %s" % diag)
            if np.any(diag > 1e6):
                raise StateEstimationFault("P对角线爆炸: max=%.2e" % np.max(diag))

        if bg is not None and np.linalg.norm(bg) > np.radians(5.0):
            raise StateEstimationFault("bg过大: %.3f rad" % np.linalg.norm(bg))
        if ba is not None and np.linalg.norm(ba) > 5.0:
            raise StateEstimationFault("ba过大: %.3f m/s²" % np.linalg.norm(ba))
        return True

    @classmethod
    def validate_solver_solution(cls, u_opt, sigma_opt, T_min, T_max):
        """检验SOCP解的合理性. 异常则抛SolverFault."""
        if u_opt is None:
            raise SolverFault("u_opt is None")
        if not np.all(np.isfinite(u_opt)):
            raise SolverFault("u_opt NaN/Inf: %s" % u_opt)

        # 推力幅值范围检查 (允许10%容差, 因归一化误差)
        u_norm = np.linalg.norm(u_opt[0]) if u_opt.ndim > 1 else np.linalg.norm(u_opt)
        if u_norm > T_max * 1.1:
            raise SolverFault("||u||=%.0fN > T_max=%.0fN" % (u_norm, T_max))
        if u_norm < 0.0:
            raise SolverFault("||u||<0: %.0fN" % u_norm)

        if sigma_opt is not None:
            if not np.all(np.isfinite(sigma_opt)):
                raise SolverFault("sigma NaN/Inf")
            if np.any(sigma_opt < T_min * 0.9):
                raise SolverFault("sigma<T_min: min=%.0f < %.0f" %
                                  (np.min(sigma_opt), T_min))
            if np.any(sigma_opt > T_max * 1.1):
                raise SolverFault("sigma>T_max: max=%.0f > %.0f" %
                                  (np.max(sigma_opt), T_max))
        return True
