"""E4: 延迟补偿验证 (Smith预估器 + 状态前推).

在E1+E3基础上加入DelayCompensator:
  EKF状态 → DelayCompensator前推150ms → G-FOLD/姿态控制

验证项:
  1. 延迟补偿后着陆精度不降级 (对比E1+E3)
  2. 闭环更平滑 (控制量抖动减小)
  3. 末端h<50m不前推 (死区模式)
  4. EKF估计误差不因前推而增大
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import src.quaternion_utils as qu
import src.rocket_params as rp
from src.dynamics import DynamicsEngine, make_state
from src.actuators import TVC, GridFin, RCS
from src.attitude_control import AttitudeController
from src.guidance import LandingGuidance
from src.wind import DrydenWind
from src.atmosphere import atmosphere
from src.flex_dynamics import FlexDynamics, OMEGA_FLEX_1, OMEGA_FLEX_2
from src.sensors import IMU, GPS, RadarAltimeter
from src.ekf import MEKF
from src.delay_comp import DelayCompensator

DT = 0.01


def run_e4(seed=42, verbose=True, use_delay_comp=True):
    """运行E4延迟补偿仿真. 返回 (success, metrics_dict)."""
    rng = np.random.default_rng(seed)
    dyn = DynamicsEngine(dt=DT)
    tvc = TVC(tau=0.05)
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

    # E4: 延迟补偿器
    delay_comp = DelayCompensator(total_delay=0.15)

    pos0 = np.array([0.0, 0.0, -2000.0])
    vel0 = np.array([0.0, 0.0, 80.0])
    state = make_state(pos_n=pos0, vel_n=vel0, q=qu.Q_VERT.copy())
    fuel = 15000.0
    t = 0.0
    M_aero_prev = np.zeros(3)

    p0_est = pos0 + rng.normal(0, 0.5, 3)
    v0_est = vel0 + rng.normal(0, 0.3, 3)
    q0_est = qu.quat_normalize(qu.Q_VERT + rng.normal(0, 0.01, 4))
    ekf = MEKF(p0_est, v0_est, q0_est, dt=DT)

    max_t = 200.0
    n_steps = int(max_t / DT)

    ekf_err_log = []
    gps_update_count = 0
    radar_update_count = 0
    terminal_mode_triggered = False

    # 存储上一步IMU测量 (供omega估计和延迟补偿)
    prev_gyro_meas = np.zeros(3)
    prev_accel_meas = np.zeros(3)

    for i in range(n_steps):
        h_true = -state[2]
        m, cg_x, I_body = rp.mass_properties(fuel)

        # === E3: 从EKF获取估计状态 ===
        state_est, sigma_est = ekf.get_state()
        # 用上一步IMU测量校正角速度 (当前步IMU还未测量)
        omega_est = ekf.get_estimated_omega(prev_gyro_meas)
        state_est_full = state_est.copy()
        state_est_full[10:13] = omega_est

        # === E4: 延迟补偿 ===
        if use_delay_comp:
            # 用上一步校正比力/角速度更新补偿器
            f_b_prev = prev_accel_meas - ekf.ba
            omega_b_prev = prev_gyro_meas - ekf.bg
            delay_comp.update_imu(f_b_prev, omega_b_prev)
            # 前推状态
            state_comp = delay_comp.compensate(state_est_full)
        else:
            state_comp = state_est_full

        # === 末端模式切换 (h<50m 禁用GPS) ===
        h_est = -state_est[2]
        if h_est < 50.0 and not terminal_mode_triggered:
            ekf.set_terminal_mode(True)
            terminal_mode_triggered = True
            if verbose:
                print("  [E4] 末端模式触发 (h<50m), 禁用GPS, 延迟补偿关闭")

        # === 制导 (用补偿后状态) ===
        throttle, q_des, omega_des, tvc_gimbal_cmd, phase = guidance.update(
            state_comp, fuel, t, DT)

        if phase == 'LANDED':
            break

        # === 姿态控制 (用补偿后状态) ===
        rho, a, _, _ = atmosphere(h_true)
        v_mag = np.linalg.norm(state[3:6])
        mach = v_mag / a if a > 0 else 0.0
        qdyn = 0.5 * rho * v_mag * v_mag

        gf_cmd, rcs_cmd, tvc_gimbal_att = att.update(
            q_des, omega_des, state_comp, mach, qdyn, cg_x, I_body, gf, rcs, DT,
            M_disturbance=M_aero_prev, phase=phase, tvc=tvc,
            thrust_actual=tvc.thrust)

        M_gf = gf.update(gf_cmd, mach, qdyn, cg_x, DT)
        M_rcs = rcs.update(rcs_cmd, cg_x, DT)

        if phase in ('G-FOLD', 'DEADBAND'):
            tvc_pitch_cmd = tvc_gimbal_att[0]
            tvc_yaw_cmd = tvc_gimbal_att[1]
        else:
            tvc_pitch_cmd = 0.0
            tvc_yaw_cmd = 0.0

        thrust_actual, gp, gy = tvc.update(
            throttle, tvc_pitch_cmd, tvc_yaw_cmd, phase, h_true, DT)

        w = wind.update(h_true, state[3:6], DT)

        # === E1: 弹性体+晃动 ===
        C_bn = qu.quat_to_rotmat(state[6:10])
        a_thrust_b = np.array([thrust_actual * np.cos(gp) * np.cos(gy),
                               thrust_actual * np.sin(gp),
                               thrust_actual * np.cos(gp) * np.sin(gy)]) / m
        a_grav_n = np.array([0, 0, rp.G0])
        a_grav_b = C_bn.T @ a_grav_n
        a_lateral_b = np.array([a_thrust_b[1] + a_grav_b[1],
                                a_thrust_b[2] + a_grav_b[2]])
        omega_b_true = state[10:13]
        thrust_b = np.array([thrust_actual * np.cos(gp) * np.cos(gy),
                             thrust_actual * np.sin(gp),
                             thrust_actual * np.cos(gp) * np.sin(gy)])
        imu_omega_dist, imu_accel_dist, aero_moment_dist, slosh_moment = \
            flex.update(DT, a_lateral_b, omega_b_true, fuel, thrust_b)

        extra_moment = M_gf + M_rcs + slosh_moment + aero_moment_dist

        # === 动力学推进 (真值) ===
        state, fuel, info = dyn.step(
            state, fuel, thrust_actual, gp, gy, w,
            extra_moment_b=extra_moment)
        M_aero_prev = info['M_aero_b'].copy()

        # === E3: 传感器测量 + EKF更新 ===
        F_thrust_b = info['F_thrust_b']
        F_aero_b = info['F_aero_b']
        f_b_true = (F_thrust_b + F_aero_b) / m
        omega_b_true = state[10:13]

        gyro_meas, accel_meas = imu.measure(
            omega_b_true + imu_omega_dist,
            f_b_true + imu_accel_dist, DT)

        # 存储本步IMU测量 (供下一步使用)
        prev_gyro_meas = gyro_meas.copy()
        prev_accel_meas = accel_meas.copy()

        # EKF预测步
        ekf.predict(gyro_meas, accel_meas, DT)

        # GPS更新
        pos_meas, vel_meas, gps_valid = gps.measure(state[0:3], state[3:6], DT)
        if gps_valid:
            ekf.update_gps(pos_meas, vel_meas)
            gps_update_count += 1

        # 雷达高度计更新
        alt_meas, radar_valid = radar.measure(state[0:3], DT)
        if radar_valid:
            ekf.update_radar(alt_meas)
            radar_update_count += 1

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
        'terminal_mode': terminal_mode_triggered,
        'ekf_log': ekf_err_log,
    }

    if verbose:
        tag = "E4(延迟补偿)" if use_delay_comp else "E4(无补偿对照)"
        print("=== %s (seed=%d) ===" % (tag, seed))
        print("着陆成功:", success)
        print("终态: h=%.2fm  vz=%.2fm/s  tilt=%.1f°  pos=%.2fm" %
              (h_final, vz_final, tilt, h_pos_err))
        print("仿真时间: %.1fs  剩余燃料: %.0fkg" % (t, fuel))
        print("\n=== EKF 估计误差 ===")
        print("位置误差: %.3fm  速度误差: %.3fm/s  姿态误差: %.3f°" %
              (pos_err_final, vel_err_final, q_err_final))
        print("GPS更新: %d  雷达更新: %d" % (gps_update_count, radar_update_count))

        print("\n=== EKF 误差时间序列 ===")
        for log in ekf_err_log[:3]:
            print("  t=%.0f pos=%.3f vel=%.3f q=%.3f°" % log[:4])
        print("  ...")
        for log in ekf_err_log[-3:]:
            print("  t=%.0f pos=%.3f vel=%.3f q=%.3f°" % log[:4])

    return success, metrics


if __name__ == '__main__':
    print("=" * 60)
    print("E4: 延迟补偿对比测试")
    print("=" * 60)

    print("\n--- 对照组: 无延迟补偿 ---")
    s1, m1 = run_e4(seed=42, verbose=True, use_delay_comp=False)

    print("\n--- 实验组: 有延迟补偿(150ms) ---")
    s2, m2 = run_e4(seed=42, verbose=True, use_delay_comp=True)

    print("\n" + "=" * 60)
    print("对比总结:")
    print("  无补偿: h=%.2f vz=%.2f tilt=%.1f° fuel=%.0f success=%s" %
          (m1['h_final'], m1['vz_final'], m1['tilt_final'], m1['fuel_final'], m1['success']))
    print("  有补偿: h=%.2f vz=%.2f tilt=%.1f° fuel=%.0f success=%s" %
          (m2['h_final'], m2['vz_final'], m2['tilt_final'], m2['fuel_final'], m2['success']))
    print("  EKF误差: 无补偿 pos=%.3f vel=%.3f  有补偿 pos=%.3f vel=%.3f" %
          (m1['pos_err_ekf'], m1['vel_err_ekf'], m2['pos_err_ekf'], m2['vel_err_ekf']))
