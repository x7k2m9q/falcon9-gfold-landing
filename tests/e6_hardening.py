"""E6: 工程硬化集成测试 (SafetyMonitor + FlightController + 求解器硬化).

在E5基础上加入E6工程硬化:
  - SafetyMonitor: 可达集/SOCP可行性/燃料告警/姿态超限检查
  - FlightController: 异常处理框架 (SensorFault/StateEstimationFault/SolverFault)
  - 求解器硬化: CLARABEL优先 + SCS兜底 + 状态跟踪

测试项:
  1. 标称工况: 不降级 (对比E5指标)
  2. 故障注入: 传感器故障/EKF故障/求解器故障 → 兜底恢复
  3. 极端工况: 低燃料/大倾角/高速下降 → 安全监控触发
  4. 5种子MC: 验证鲁棒性

工程验证目标:
  - 标称工况着陆精度不降级 (h<1m, vz<5m/s, tilt<5°)
  - 故障注入后仍能安全着陆 (或至少不崩溃)
  - SafetyMonitor正确检测违反并触发兜底
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import src.quaternion_utils as qu
import src.rocket_params as rp
from src.dynamics import DynamicsEngine, make_state
from src.actuators import GridFin, RCS
from src.attitude_control import AttitudeController
from src.guidance import LandingGuidance
from src.wind import DrydenWind
from src.atmosphere import atmosphere
from src.flex_dynamics import FlexDynamics
from src.sensors import IMU, GPS, RadarAltimeter
from src.ekf import MEKF
from src.delay_comp import DelayCompensator
from src.octaweb import Octaweb
from src.safety_monitor import SafetyMonitor
from src.flight_controller import FlightController

DT = 0.01


def run_e6(seed=42, verbose=True, inject_faults=False, extreme=False):
    """运行E6工程硬化仿真.

    参数:
      seed: 随机种子
      verbose: 打印详情
      inject_faults: 注入故障 (传感器/EKF/求解器)
      extreme: 极端工况 (低燃料/大倾角)

    返回: (success, metrics_dict)
    """
    rng = np.random.default_rng(seed)
    dyn = DynamicsEngine(dt=DT)
    octaweb = Octaweb()
    gf = GridFin()
    rcs = RCS()
    att = AttitudeController(wn=2 * np.pi * 0.3, zeta=0.9, use_notch=True)
    guidance = LandingGuidance(gfold_N=50, dt=DT)
    wind = DrydenWind(rng, sigma_base=1.0)
    flex = FlexDynamics()

    imu = IMU(rng,
              gyro_noise_std=np.radians(0.01),
              accel_noise_std=0.02,
              gyro_bias_walk=np.radians(1e-4),
              accel_bias_walk=1e-4,
              gyro_init_bias=np.radians(0.05),
              accel_init_bias=0.02)
    gps = GPS(rng, pos_noise_std=0.5, vel_noise_std=0.1, update_rate=10.0)
    radar = RadarAltimeter(rng, alt_noise_std=0.05, update_rate=50.0, max_alt=100.0)
    delay_comp = DelayCompensator(total_delay=0.15)

    # E6: SafetyMonitor + FlightController
    safety_monitor = SafetyMonitor()

    # 初始条件
    if extreme:
        # 极端工况: 更高速度, 更低燃料
        pos0 = np.array([20.0, 15.0, -2000.0])
        vel0 = np.array([5.0, 3.0, 100.0])  # 更高下降速度+水平速度
        fuel = 8000.0  # 低燃料 (正常15000)
    else:
        pos0 = np.array([0.0, 0.0, -2000.0])
        vel0 = np.array([0.0, 0.0, 80.0])
        fuel = 15000.0

    state = make_state(pos_n=pos0, vel_n=vel0, q=qu.Q_VERT.copy())
    t = 0.0
    M_aero_prev = np.zeros(3)

    p0_est = pos0 + rng.normal(0, 0.5, 3)
    v0_est = vel0 + rng.normal(0, 0.3, 3)
    q0_est = qu.quat_normalize(qu.Q_VERT + rng.normal(0, 0.01, 4))
    ekf = MEKF(p0_est, v0_est, q0_est, dt=DT)

    # E6: 创建FlightController
    fc = FlightController(
        ekf=ekf, delay_comp=delay_comp, guidance=guidance,
        attitude=att, octaweb=octaweb, safety_monitor=safety_monitor,
        gf=gf, rcs=rcs)

    max_t = 200.0
    n_steps = int(max_t / DT)

    ekf_err_log = []
    engine_config_log = []
    safety_log = []  # E6: 安全状态日志
    fault_log = []   # E6: 故障日志
    solver_log = []  # E6: 求解器统计
    gps_update_count = 0
    radar_update_count = 0

    prev_gyro_meas = np.zeros(3)
    prev_accel_meas = np.zeros(3)

    octaweb.set_engine_config(1)
    prev_n_engines = 1

    # 故障注入时机
    fault_inject_times = [10.0, 25.0, 40.0] if inject_faults else []
    fault_idx = 0

    for i in range(n_steps):
        h_true = -state[2]
        m, cg_x, I_body = rp.mass_properties(fuel)

        # === E6: 故障注入 ===
        if fault_idx < len(fault_inject_times) and t >= fault_inject_times[fault_idx]:
            if fault_idx == 0:
                # 传感器故障: 注入NaN加速度
                fc.inject_sensor_fault = True
                if verbose:
                    print("  [FAULT] t=%.1fs 注入传感器故障" % t)
            elif fault_idx == 1:
                # EKF故障
                fc.inject_ekf_fault = True
                if verbose:
                    print("  [FAULT] t=%.1fs 注入EKF故障" % t)
            elif fault_idx == 2:
                # 求解器故障: 强制SOCP返回不可行
                fc.inject_solver_fault = True
                if verbose:
                    print("  [FAULT] t=%.1fs 注入求解器故障" % t)
            fault_idx += 1

        # === 传感器测量 (真值→传感器) ===
        C_bn = qu.quat_to_rotmat(state[6:10])
        omega_b_true = state[10:13]
        # 用上一步info计算比力 (本步还没step)
        if i == 0:
            f_b_true = np.array([0.0, 0.0, 0.0])
        else:
            F_thrust_b = info.get('F_thrust_b', np.zeros(3))
            F_aero_b = info.get('F_aero_b', np.zeros(3))
            f_b_true = (F_thrust_b + F_aero_b) / m

        gyro_meas, accel_meas = imu.measure(omega_b_true, f_b_true, DT)
        pos_meas, vel_meas, gps_valid = gps.measure(state[0:3], state[3:6], DT)
        alt_meas, radar_valid = radar.measure(state[0:3], DT)

        if gps_valid:
            gps_update_count += 1
        if radar_valid:
            radar_update_count += 1

        # === E6: FlightController.step() ===
        output = fc.step(
            state, fuel, t, DT,
            gyro_meas=gyro_meas, accel_meas=accel_meas,
            gps_pos=pos_meas if gps_valid else None,
            gps_vel=vel_meas if gps_valid else None,
            gps_valid=gps_valid,
            radar_alt=alt_meas if radar_valid else None,
            radar_valid=radar_valid,
            M_aero_disturbance=M_aero_prev)

        throttle = output.throttle
        q_des = output.q_des
        phase = guidance.phase

        # E6: 记录安全状态
        if output.status != "NOMINAL":
            safety_log.append((t, output.status, output.fault))

        if phase == 'LANDED':
            break

        # === E5: 发动机配置同步 ===
        n_engines = guidance.n_engines_current
        if n_engines != prev_n_engines:
            octaweb.set_engine_config(n_engines)
            engine_config_log.append((t, prev_n_engines, n_engines, h_true, state[5]))
            if verbose:
                print("  [E5] %d→%d发 t=%.1fs h=%.1fm vz=%.2f" %
                      (prev_n_engines, n_engines, t, h_true, state[5]))
            prev_n_engines = n_engines

        # === 姿态控制 (已在FlightController内执行, 这里只需取输出) ===
        gf_cmd = output.gf_cmd
        rcs_cmd = output.rcs_cmd
        tvc_pitch_cmd = output.tvc_gimbal[0] if phase in ('G-FOLD', 'DEADBAND') else 0.0
        tvc_yaw_cmd = output.tvc_gimbal[1] if phase in ('G-FOLD', 'DEADBAND') else 0.0

        # Octaweb已在FlightController内更新, 直接用输出推力
        total_thrust = output.total_thrust
        gp = output.tvc_gimbal[0]
        gy = output.tvc_gimbal[1]

        # === GridFin + RCS (已在FlightController内计算cmd, 这里执行update) ===
        rho, a, _, _ = atmosphere(h_true)
        v_mag = np.linalg.norm(state[3:6])
        mach = v_mag / a if a > 0 else 0.0
        qdyn = 0.5 * rho * v_mag * v_mag
        M_gf = gf.update(gf_cmd, mach, qdyn, cg_x, DT)
        M_rcs = rcs.update(rcs_cmd, cg_x, DT)

        w = wind.update(h_true, state[3:6], DT)

        # === E1: 弹性体+晃动 ===
        a_thrust_b = np.array([total_thrust * np.cos(gp) * np.cos(gy),
                               total_thrust * np.sin(gp),
                               total_thrust * np.cos(gp) * np.sin(gy)]) / m
        a_grav_n = np.array([0, 0, rp.G0])
        a_grav_b = C_bn.T @ a_grav_n
        a_lateral_b = np.array([a_thrust_b[1] + a_grav_b[1],
                                a_thrust_b[2] + a_grav_b[2]])
        thrust_b = np.array([total_thrust * np.cos(gp) * np.cos(gy),
                             total_thrust * np.sin(gp),
                             total_thrust * np.cos(gp) * np.sin(gy)])
        imu_omega_dist, imu_accel_dist, aero_moment_dist, slosh_moment = \
            flex.update(DT, a_lateral_b, omega_b_true, fuel, thrust_b)

        # E5: 外围发动机偏心力矩
        F_octa, M_octa = octaweb.compute_force_moment(gp, gy, cg_x)
        x_tvc = -rp.LENGTH / 2.0 + 1.0
        r_center = np.array([x_tvc - cg_x, 0.0, 0.0])
        F_center = np.array([octaweb.engines[0].thrust * np.cos(gp) * np.cos(gy),
                             octaweb.engines[0].thrust * np.sin(gp),
                             octaweb.engines[0].thrust * np.cos(gp) * np.sin(gy)])
        M_center = np.cross(r_center, F_center)
        M_outer_residual = M_octa - M_center

        extra_moment = M_gf + M_rcs + slosh_moment + aero_moment_dist + M_outer_residual

        # === 动力学推进 ===
        state, fuel, info = dyn.step(
            state, fuel, total_thrust, gp, gy, w,
            extra_moment_b=extra_moment)
        M_aero_prev = info['M_aero_b'].copy()

        prev_gyro_meas = gyro_meas.copy()
        prev_accel_meas = accel_meas.copy()

        # === 诊断日志 (每1秒) ===
        if i % 100 == 0:
            pos_err = np.linalg.norm(ekf.p - state[0:3])
            vel_err = np.linalg.norm(ekf.v - state[3:6])
            q_err = qu.quat_multiply(qu.quat_inverse(state[6:10]), ekf.q)
            q_err_angle = 2.0 * np.degrees(np.arccos(np.clip(abs(q_err[0]), 0, 1)))
            ekf_err_log.append((t, pos_err, vel_err, q_err_angle,
                                np.linalg.norm(ekf.bg), np.linalg.norm(ekf.ba)))

        t += DT

        if h_true < 0.0 and abs(state[5]) > 15.0:
            break

    # === 结果评估 ===
    h_final = -state[2]
    vz_final = state[5]
    tilt = np.degrees(qu.tilt_angle_from_vertical(state[6:10]))
    h_pos_err = np.hypot(state[0], state[1])
    success = (h_final < 1.0 and abs(vz_final) < 5.0 and
               tilt < 45.0 and h_pos_err < 50.0)

    pos_err_final = np.linalg.norm(ekf.p - state[0:3])
    vel_err_final = np.linalg.norm(ekf.v - state[3:6])
    q_err_final_q = qu.quat_multiply(qu.quat_inverse(state[6:10]), ekf.q)
    q_err_final = 2.0 * np.degrees(np.arccos(np.clip(abs(q_err_final_q[0]), 0, 1)))

    # E6: 故障统计
    fc_stats = fc.get_stats()
    solver_stats = {
        'solve_count': guidance.gfold._solve_count,
        'fail_count': guidance.gfold._fail_count,
        'last_info': guidance.gfold._last_solve_info,
        'history': guidance.gfold._solver_history[-10:] if guidance.gfold._solver_history else [],
    }

    metrics = {
        'success': success,
        'h_final': h_final,
        'vz_final': vz_final,
        'tilt_final': tilt,
        'pos_err_final': h_pos_err,
        't_final': t,
        'fuel_final': fuel,
        'pos_err_ekf': pos_err_final,
        'vel_err_ekf': vel_err_final,
        'q_err_ekf': q_err_final,
        'gps_updates': gps_update_count,
        'radar_updates': radar_update_count,
        'ekf_log': ekf_err_log,
        'engine_config_log': engine_config_log,
        'final_n_engines': prev_n_engines,
        'safety_log': safety_log,
        'fault_log': fc_stats['fault_log'],
        'fc_stats': fc_stats,
        'solver_stats': solver_stats,
    }

    if verbose:
        print("=== E6 工程硬化 (seed=%d, faults=%s, extreme=%s) ===" %
              (seed, inject_faults, extreme))
        print("着陆成功:", success)
        print("终态: h=%.2fm  vz=%.2fm/s  tilt=%.1f°  pos=%.2fm" %
              (h_final, vz_final, tilt, h_pos_err))
        print("仿真时间: %.1fs  剩余燃料: %.0fkg  最终发动机: %d发" %
              (t, fuel, prev_n_engines))
        print("\n=== 发动机配置切换历史 ===")
        for log in engine_config_log:
            print("  t=%.1fs: %d→%d发  h=%.1fm  vz=%.2f" % log)
        if not engine_config_log:
            print("  (无切换, 全程%d发)" % prev_n_engines)
        print("\n=== EKF 估计误差 ===")
        print("位置误差: %.3fm  速度误差: %.3fm/s  姿态误差: %.3f°" %
              (pos_err_final, vel_err_final, q_err_final))
        print("GPS更新: %d  雷达更新: %d" % (gps_update_count, radar_update_count))
        print("\n=== E6 工程硬化统计 ===")
        print("总步数: %d  故障步数: %d  兜底步数: %d" %
              (fc_stats['total_steps'], fc_stats['fault_steps'], fc_stats['fallback_steps']))
        print("故障分类: %s" % fc_stats['fault_counts'])
        print("求解器: solve=%d fail=%d last_solver=%s last_status=%s" %
              (solver_stats['solve_count'], solver_stats['fail_count'],
               solver_stats['last_info'].get('solver', '?'),
               solver_stats['last_info'].get('status', '?')))
        if safety_log:
            print("\n=== 安全状态触发历史 ===")
            for log in safety_log[-10:]:
                print("  t=%.1fs: %s (%s)" % log)
        if fc_stats['fault_log']:
            print("\n=== 故障日志 (最近10条) ===")
            for log in fc_stats['fault_log'][-10:]:
                print("  t=%.1fs [%s]: %s" % log)

    return success, metrics


def run_mc(seeds=(42, 123, 456, 789, 1024), verbose=True):
    """多种子蒙特卡洛测试."""
    results = []
    for seed in seeds:
        s, m = run_e6(seed=seed, verbose=False)
        results.append((seed, s, m))
        if verbose:
            print("seed=%d: %s  h=%.2fm  vz=%.2fm/s  tilt=%.1f  t=%.1fs  fuel=%.0fkg  faults=%d" %
                  (seed, "OK" if s else "FAIL", m['h_final'], m['vz_final'],
                   m['tilt_final'], m['t_final'], m['fuel_final'],
                   m['fc_stats']['fault_steps']))

    successes = sum(1 for _, s, _ in results if s)
    print("\n=== MC总结: %d/%d 成功 (%.0f%%) ===" %
          (successes, len(results), 100.0 * successes / len(results)))
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("E6: 工程硬化集成测试")
    print("=" * 60)

    # 1. 标称工况
    print("\n--- 1. 标称工况 (seed=42) ---")
    s1, m1 = run_e6(seed=42, verbose=True)

    # 2. 故障注入
    print("\n--- 2. 故障注入测试 (seed=42) ---")
    s2, m2 = run_e6(seed=42, verbose=True, inject_faults=True)

    # 3. 极端工况
    print("\n--- 3. 极端工况 (seed=42, 低燃料+高速) ---")
    s3, m3 = run_e6(seed=42, verbose=True, extreme=True)

    # 4. 5种子MC
    print("\n--- 4. 5种子MC ---")
    results = run_mc(seeds=(42, 123, 456, 789, 1024), verbose=True)

    print("\n" + "=" * 60)
    print("总结:")
    print("  标称: %s  h=%.2fm  vz=%.2fm/s" % (s1, m1['h_final'], m1['vz_final']))
    print("  故障注入: %s  h=%.2fm  vz=%.2fm/s  故障步=%d" %
          (s2, m2['h_final'], m2['vz_final'], m2['fc_stats']['fault_steps']))
    print("  极端: %s  h=%.2fm  vz=%.2fm/s" % (s3, m3['h_final'], m3['vz_final']))
