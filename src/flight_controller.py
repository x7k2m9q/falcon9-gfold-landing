"""E6: 飞行控制器异常处理框架.

理论方案2.0 E6.3节:
  封装EKF+延迟补偿+制导+姿态控制+控制分配为统一step()接口.
  分级异常处理:
    SensorFault → 用上一拍传感器数据, 标记故障
    StateEstimationFault → EKF复位到上一好态, 协方差膨胀
    SolverFault → 退化到PD控制 (hover-slam bang-bang)
    SafetyViolation → 根据违反类型选择兜底策略
    未捕获异常 → safe_mode (满推力+垂直姿态)

设计原则 (工程直觉):
  1. 不破坏现有E5架构, FlightController是可选的包装层
  2. 异常处理是"最后一道防线", 不应频繁触发
  3. 兜底策略必须物理可行 (不能让T_min>重力的发动机悬停)
  4. 故障计数与恢复: 短期故障(1-2拍)用上一拍数据, 长期故障(>3拍)切换兜底

控制流:
  sensors → validate → EKF predict/update → validate → delay_comp
  → safety_check → guidance/fallback → attitude → allocator → output

输出: ControlOutput(throttle, q_des, tvc_gimbal, gf_cmd, rcs_cmd, status, fault)
"""
import numpy as np
from . import rocket_params as rp
from . import quaternion_utils as qu
from .safety_monitor import (
    SafetyMonitor, InputValidator,
    SensorFault, StateEstimationFault, SolverFault, SafetyViolation, FlightException
)


class ControlOutput:
    """控制输出容器."""
    def __init__(self, throttle=0.0, q_des=None, omega_des=None,
                 tvc_gimbal=None, gf_cmd=None, rcs_cmd=None,
                 status="NOMINAL", fault="", n_engines=1, total_thrust=0.0):
        self.throttle = throttle
        self.q_des = q_des if q_des is not None else qu.Q_VERT.copy()
        self.omega_des = omega_des if omega_des is not None else np.zeros(3)
        self.tvc_gimbal = tvc_gimbal if tvc_gimbal is not None else np.zeros(2)
        self.gf_cmd = gf_cmd if gf_cmd is not None else np.zeros(3)  # [roll,pitch,yaw]
        self.rcs_cmd = rcs_cmd if rcs_cmd is not None else np.zeros(3)  # [Mx,My,Mz]
        self.status = status
        self.fault = fault
        self.n_engines = n_engines
        self.total_thrust = total_thrust  # E6: Octaweb总推力 (含关机尾推)


class FallbackPD:
    """PD兜底控制器 (无SOCP, 纯反馈).

    当SOCP不可行时使用. 基于hover-slam剖面的bang-bang控制:
      target_vz = 2*sqrt(h)  (恒定减速度2m/s²)
      vz > target+deadband: throttle = T_min + Kp*(vz-target)
      vz < target-deadband: throttle = 0 (自由落体省油)
      中间: 保持上一步 (滞回防抖)

    水平修正: 小倾角PD (最大8°), 对抗水平偏差.
    """
    def __init__(self):
        self.last_throttle = rp.THRUST_MIN_PCT
        self.Kp = 0.12
        self.deadband = 0.5

    def update(self, state, m, n_engines=1):
        from .guidance import q_des_from_thrust_dir
        pos_n = state[0:3]
        vel_n = state[3:6]
        h = -pos_n[2]
        vz = vel_n[2]

        # hover-slam剖面
        target_vz = max(3.0, 2.0 * np.sqrt(max(h, 0.0)))
        if vz > target_vz + self.deadband:
            throttle = np.clip(rp.THRUST_MIN_PCT + self.Kp * (vz - target_vz),
                               rp.THRUST_MIN_PCT, 1.0)
            self.last_throttle = throttle
        elif vz < target_vz - self.deadband:
            throttle = 0.0
            self.last_throttle = 0.0
        else:
            throttle = self.last_throttle

        # 水平修正 (最大8°)
        px, py = pos_n[0], pos_n[1]
        vx, vy = vel_n[0], vel_n[1]
        a_h = -0.08 * np.array([px, py]) - 0.5 * np.array([vx, vy])
        a_mag = np.linalg.norm(a_h)
        max_a = rp.G0 * np.sin(np.radians(8.0))
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

        return throttle, q_des, np.zeros(3)


class FlightController:
    """飞行控制器: 封装完整控制管线 + 异常处理.

    使用方式:
      fc = FlightController(ekf, delay_comp, guidance, att, octaweb, ...)
      for each step:
          output = fc.step(sensors, fuel, t, dt)
          # 将output应用到dynamics

    异常处理层次 (由内到外):
      1. validate_sensors → SensorFault → fallback_sensor (用上一拍数据)
      2. EKF update → NaN/Inf → StateEstimationFault → fallback_ekf_reset
      3. G-FOLD solve → infeasible → SolverFault → fallback_pd
      4. SafetyMonitor → SafetyViolation → 对应兜底策略
      5. 未捕获异常 → fallback_safe_mode (满推力+垂直)
    """

    def __init__(self, ekf, delay_comp, guidance, attitude, octaweb,
                 safety_monitor=None, gf=None, rcs=None):
        self.ekf = ekf
        self.delay_comp = delay_comp
        self.guidance = guidance
        self.attitude = attitude
        self.octaweb = octaweb
        self.gf = gf
        self.rcs = rcs
        self.safety = safety_monitor if safety_monitor is not None else SafetyMonitor()

        # 兜底控制器
        self.fallback_pd = FallbackPD()

        # 故障状态跟踪
        self.fault_counts = {
            'sensor': 0, 'estimation': 0, 'solver': 0, 'safety': 0, 'unknown': 0
        }
        self.fault_threshold = 5  # 连续5拍故障后升级处理
        self.last_good_state = None  # 上一好态 (用于EKF复位)
        self.last_good_sensors = None  # 上一好传感器数据
        self.last_good_throttle = 0.5
        self.last_good_q_des = qu.Q_VERT.copy()

        # 末端模式
        self.terminal_mode = False
        self.terminal_alt = 50.0

        # 上一拍IMU (用于延迟补偿)
        self.prev_gyro_meas = np.zeros(3)
        self.prev_accel_meas = np.zeros(3)

        # 故障注入接口 (测试用)
        self.inject_sensor_fault = False
        self.inject_solver_fault = False
        self.inject_ekf_fault = False

        # 统计
        self.total_steps = 0
        self.fault_steps = 0
        self.fallback_steps = 0
        self.fault_log = []

    def step(self, true_state, fuel, t, dt,
             gyro_meas=None, accel_meas=None,
             gps_pos=None, gps_vel=None, gps_valid=False,
             radar_alt=None, radar_valid=False,
             M_aero_disturbance=None):
        """执行一拍控制.

        参数:
          true_state: 真值状态 (13D, 仅用于传感器生成已在外部完成)
          fuel: 剩余燃料 kg
          t, dt: 时间
          gyro_meas, accel_meas: IMU测量 (已含噪声)
          gps_pos, gps_vel, gps_valid: GPS测量
          radar_alt, radar_valid: 雷达测量

        返回: ControlOutput
        """
        self.total_steps += 1
        fault_msg = ""

        try:
            # === 1. 传感器检验 ===
            if gyro_meas is None or accel_meas is None:
                raise SensorFault("IMU数据缺失")

            # 故障注入 (测试用)
            if self.inject_sensor_fault:
                self.inject_sensor_fault = False  # 单次注入
                raise SensorFault("注入传感器故障")

            InputValidator.validate_sensor(gyro_meas, accel_meas,
                                           gps_pos if gps_valid else None,
                                           gps_vel if gps_valid else None,
                                           radar_alt if radar_valid else None)

            # 传感器OK, 记录为last_good
            self.last_good_sensors = (gyro_meas.copy(), accel_meas.copy())
            self.fault_counts['sensor'] = 0

            # === 2. EKF预测 ===
            self.ekf.predict(gyro_meas, accel_meas, dt)

            # === 3. EKF更新 (GPS/Radar) ===
            if gps_valid:
                self.ekf.update_gps(gps_pos, gps_vel)
            if radar_valid:
                self.ekf.update_radar(radar_alt)

            # === 4. EKF状态检验 ===
            if self.inject_ekf_fault:
                self.inject_ekf_fault = False
                raise StateEstimationFault("注入EKF故障")

            state_est, sigma_est = self.ekf.get_state()
            omega_est = self.ekf.get_estimated_omega(self.prev_gyro_meas)
            state_est_full = state_est.copy()
            state_est_full[10:13] = omega_est

            InputValidator.validate_ekf_state(
                self.ekf.p, self.ekf.v, self.ekf.q,
                self.ekf.bg, self.ekf.ba, self.ekf.P)

            # EKF OK, 记录last_good
            self.last_good_state = state_est_full.copy()
            self.fault_counts['estimation'] = 0

            # === 5. 延迟补偿 ===
            f_b_prev = self.prev_accel_meas - self.ekf.ba
            omega_b_prev = self.prev_gyro_meas - self.ekf.bg
            self.delay_comp.update_imu(f_b_prev, omega_b_prev)
            state_comp = self.delay_comp.compensate(state_est_full)

            # 末端模式
            h_est = -state_est[2]
            if h_est < self.terminal_alt and not self.terminal_mode:
                self.ekf.set_terminal_mode(True)
                self.terminal_mode = True

            # === 6. 安全检查 ===
            n_engines = self.guidance.n_engines_current
            status, violation = self.safety.check(state_comp, fuel, n_engines=n_engines)

            # === 7. 制导 ===
            if status == SafetyMonitor.NOMINAL:
                # 故障注入 (测试用): 求解器故障
                if self.inject_solver_fault:
                    self.inject_solver_fault = False
                    self.safety.update_solve_status("injected_fault", False)
                    raise SolverFault("注入求解器故障")

                # 正常制导
                throttle, q_des, omega_des, tvc_gimbal_cmd, phase = self.guidance.update(
                    state_comp, fuel, t, dt)

                # 检查制导输出
                if not np.isfinite(throttle) or throttle < 0 or throttle > 1.5:
                    raise SolverFault("throttle异常: %.3f" % throttle)
                if not np.all(np.isfinite(q_des)):
                    raise SolverFault("q_des NaN/Inf")

                # 更新安全监控的求解器状态 (仅在本步有求解时更新, 避免重复计数)
                if getattr(self.guidance, 'solve_happened_this_step', False):
                    solve_info = getattr(self.guidance.gfold, '_last_solve_info', None)
                    if solve_info is not None:
                        self.safety.update_solve_status(
                            solve_info.get('status', 'optimal'),
                            solve_info.get('success', True))

            else:
                # 安全违反 → 兜底控制
                fault_msg = "Safety: %s (%s)" % (status, violation)
                self.fault_log.append((t, "safety", fault_msg))
                self.fallback_steps += 1

                if status == SafetyMonitor.FALLBACK_PD:
                    # SOCP不可行 → PD兜底
                    m, cg_x, I_body = rp.mass_properties(fuel)
                    throttle, q_des, omega_des = self.fallback_pd.update(
                        state_comp, m, n_engines)
                    phase = self.guidance.phase
                    tvc_gimbal_cmd = np.zeros(2)
                else:
                    # ABORT_MAX_BRAKE / EMERGENCY_RECOVER / MIN_FUEL_GLIDE
                    fb_throttle = self.safety.get_fallback_throttle(
                        status, -state_comp[2], state_comp[5], n_engines)
                    throttle = fb_throttle if fb_throttle is not None else rp.THRUST_MIN_PCT
                    q_des = qu.Q_VERT.copy()  # 紧急情况强制垂直
                    omega_des = np.zeros(3)
                    phase = self.guidance.phase
                    tvc_gimbal_cmd = np.zeros(2)

            # === 8. 姿态控制 ===
            h_true = -true_state[2]
            from .atmosphere import atmosphere
            rho, a, _, _ = atmosphere(h_true)
            v_mag = np.linalg.norm(true_state[3:6])
            mach = v_mag / a if a > 0 else 0.0
            qdyn = 0.5 * rho * v_mag * v_mag
            m, cg_x, I_body = rp.mass_properties(fuel)

            # M_aero_disturbance由外部传入 (上一步气动矩, 用于前馈补偿)
            M_dist = M_aero_disturbance if M_aero_disturbance is not None else np.zeros(3)
            gf_cmd, rcs_cmd, tvc_gimbal_att = self.attitude.update(
                q_des, omega_des, state_comp, mach, qdyn, cg_x, I_body,
                self.gf, self.rcs, dt,
                M_disturbance=M_dist, phase=phase,
                tvc=self.octaweb.center_tvc,
                thrust_actual=self.octaweb.center_tvc.thrust)

            # TVC指令选择: G-FOLD/DEADBAND用姿态环输出
            if phase in ('G-FOLD', 'DEADBAND'):
                tvc_pitch_cmd = tvc_gimbal_att[0]
                tvc_yaw_cmd = tvc_gimbal_att[1]
            else:
                tvc_pitch_cmd = 0.0
                tvc_yaw_cmd = 0.0

            # === 9. 控制分配 (Octaweb) ===
            n_engines_new = self.guidance.n_engines_current
            if n_engines_new != self.octaweb.n_active:
                self.octaweb.set_engine_config(n_engines_new)

            total_thrust, gp, gy = self.octaweb.update(
                throttle, tvc_pitch_cmd, tvc_yaw_cmd,
                phase, h_true, dt)

            # 记录last_good
            self.last_good_throttle = throttle
            self.last_good_q_des = q_des.copy()
            self.prev_gyro_meas = gyro_meas.copy()
            self.prev_accel_meas = accel_meas.copy()

            return ControlOutput(
                throttle=throttle, q_des=q_des, omega_des=omega_des,
                tvc_gimbal=np.array([gp, gy]), gf_cmd=gf_cmd, rcs_cmd=rcs_cmd,
                status=status, fault=fault_msg, n_engines=n_engines_new,
                total_thrust=total_thrust)

        except SensorFault as e:
            return self._handle_sensor_fault(e, true_state, fuel, t, dt)
        except StateEstimationFault as e:
            return self._handle_estimation_fault(e, true_state, fuel, t, dt)
        except SolverFault as e:
            return self._handle_solver_fault(e, true_state, fuel, t, dt)
        except SafetyViolation as e:
            return self._handle_safety_violation(e, true_state, fuel, t, dt)
        except Exception as e:
            return self._handle_unknown_fault(e, true_state, fuel, t, dt)

    # ============================================================
    # 异常处理方法
    # ============================================================
    def _handle_sensor_fault(self, e, true_state, fuel, t, dt):
        """传感器故障: 用上一拍传感器数据."""
        self.fault_counts['sensor'] += 1
        self.fault_steps += 1
        self.fault_log.append((t, "sensor", str(e)))

        if self.last_good_sensors is not None and self.fault_counts['sensor'] < self.fault_threshold:
            # 短期故障: 用上一拍数据重试
            gyro, accel = self.last_good_sensors
            return self.step(true_state, fuel, t, dt,
                             gyro_meas=gyro, accel_meas=accel,
                             gps_valid=False, radar_valid=False)
        else:
            # 长期故障: safe_mode
            return self._fallback_safe_mode(true_state, fuel, t, dt, "sensor_fault_persistent")

    def _handle_estimation_fault(self, e, true_state, fuel, t, dt):
        """EKF故障: 用PD兜底一步, 不复位EKF (让EKF自然恢复).

        工程决策: EKF复位风险大于收益. 复位后P矩阵和零偏需要时间收敛,
        期间估计可能发散. 更安全的做法是: 用PD兜底一步, EKF继续运行,
        靠下一拍的IMU+GPS+Radar自然修正.
        """
        self.fault_counts['estimation'] += 1
        self.fault_steps += 1
        self.fault_log.append((t, "estimation", str(e)))

        # 不复位EKF, 只用PD兜底这一步. EKF继续运行, 下一拍自然恢复.
        return self._fallback_pd_output(true_state, fuel, t, dt, "ekf_fault_skip")

    def _handle_solver_fault(self, e, true_state, fuel, t, dt):
        """求解器故障: PD兜底."""
        self.fault_counts['solver'] += 1
        self.fault_steps += 1
        self.fault_log.append((t, "solver", str(e)))

        self.safety.update_solve_status("solver_fault", False)
        return self._fallback_pd_output(true_state, fuel, t, dt, "solver_fault")

    def _handle_safety_violation(self, e, true_state, fuel, t, dt):
        """安全违反: 根据类型兜底."""
        self.fault_counts['safety'] += 1
        self.fault_steps += 1
        self.fault_log.append((t, "safety", str(e)))
        return self._fallback_safe_mode(true_state, fuel, t, dt, "safety_violation")

    def _handle_unknown_fault(self, e, true_state, fuel, t, dt):
        """未捕获异常: safe_mode."""
        self.fault_counts['unknown'] += 1
        self.fault_steps += 1
        self.fault_log.append((t, "unknown", str(e)))
        return self._fallback_safe_mode(true_state, fuel, t, dt, "unknown_fault")

    def _fallback_pd_output(self, true_state, fuel, t, dt, reason):
        """PD兜底输出 (用真值, 工程上应替换为纯惯导)."""
        m, cg_x, I_body = rp.mass_properties(fuel)
        n_engines = self.guidance.n_engines_current
        throttle, q_des, omega_des = self.fallback_pd.update(true_state, m, n_engines)
        # 执行Octaweb更新 (fallback模式TVC=0, 垂直推力)
        h_true = -true_state[2]
        if n_engines != self.octaweb.n_active:
            self.octaweb.set_engine_config(n_engines)
        total_thrust, gp, gy = self.octaweb.update(
            throttle, 0.0, 0.0, self.guidance.phase, h_true, dt)
        return ControlOutput(
            throttle=throttle, q_des=q_des, omega_des=omega_des,
            tvc_gimbal=np.array([gp, gy]),
            status="FALLBACK_PD", fault=reason, n_engines=n_engines,
            total_thrust=total_thrust)

    def _fallback_safe_mode(self, true_state, fuel, t, dt, reason):
        """安全模式: 满推力+垂直姿态 (最后防线)."""
        h = -true_state[2]
        vz = true_state[5]
        # 低空且慢速 → 关机 (避免飞走)
        if h < 5.0 and abs(vz) < 3.0:
            throttle = 0.0
        else:
            throttle = 1.0  # 满推力刹车
        # 执行Octaweb更新
        if 1 != self.octaweb.n_active:
            self.octaweb.set_engine_config(1)
        total_thrust, gp, gy = self.octaweb.update(
            throttle, 0.0, 0.0, self.guidance.phase, h, dt)
        return ControlOutput(
            throttle=throttle, q_des=qu.Q_VERT.copy(), omega_des=np.zeros(3),
            tvc_gimbal=np.array([gp, gy]),
            status="SAFE_MODE", fault=reason, n_engines=1,
            total_thrust=total_thrust)

    def get_stats(self):
        """返回故障统计."""
        return {
            'total_steps': self.total_steps,
            'fault_steps': self.fault_steps,
            'fallback_steps': self.fallback_steps,
            'fault_counts': self.fault_counts.copy(),
            'fault_log': self.fault_log.copy(),
            'fault_rate': self.fault_steps / max(1, self.total_steps),
        }
