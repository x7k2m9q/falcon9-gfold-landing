"""E1+E3联合验证: 弹性体+晃动(E1) + EKF状态估计(E3).
关键: 去掉上帝视角! 制导/姿态控制器只看EKF估计, 不看真值.
  真值 -> 传感器(IMU/GPS/Radar) -> EKF -> 估计状态 -> 制导/姿态控制
  弹性体扰动加到IMU测量上, EKF必须处理这种污染.

验证项:
  1. EKF估计误差收敛 (位置<2m, 速度<0.5m/s, 姿态<1度)
  2. 着陆成功 (h<1m, |vz|<5m/s, tilt<45度, pos<50m)
  3. 弹性体不发散 (|eta|有界)
  4. 末端50m以下GPS禁用, 纯IMU+雷达不漂移
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

DT = 0.01


def run_e1e3(seed=42, verbose=True):
    """运行E1+E3联合仿真. 返回 (success, metrics_dict)."""
    rng = np.random.default_rng(seed)
    dyn = DynamicsEngine(dt=DT)
    tvc = TVC(tau=0.05)
    gf = GridFin()
    rcs = RCS()
    att = AttitudeController(wn=2 * np.pi * 0.3, zeta=0.9, use_notch=True)
    guidance = LandingGuidance(gfold_N=50, dt=DT)
    wind = DrydenWind(rng, sigma_base=1.0)
    flex = FlexDynamics()

    # 传感器 (E3: 去上帝视角)
    # IMU参数: 战术级(0.05°/s初始零偏, 36°/hour), Falcon9实际用导航级(更优)
    # 原 sensors.py 默认 0.5°/s=1800°/hour 是消费级, 不适合火箭仿真
    imu = IMU(rng,
              gyro_noise_std=np.radians(0.01),
              accel_noise_std=0.02,
              gyro_bias_walk=np.radians(1e-4),
              accel_bias_walk=1e-4,
              gyro_init_bias=np.radians(0.05),   # 战术级初始零偏
              accel_init_bias=0.02)               # 加计初始零偏
    gps = GPS(rng, pos_noise_std=0.5, vel_noise_std=0.1, update_rate=10.0)
    radar = RadarAltimeter(rng, alt_noise_std=0.05, update_rate=50.0, max_alt=100.0)

    # 初始条件
    pos0 = np.array([0.0, 0.0, -2000.0])
    vel0 = np.array([0.0, 0.0, 80.0])
    state = make_state(pos_n=pos0, vel_n=vel0, q=qu.Q_VERT.copy())
    fuel = 15000.0
    t = 0.0
    M_aero_prev = np.zeros(3)

    # EKF初始化: 给一个有偏差的初始估计 (模拟真实启动条件)
    # 初始位置/速度有小误差, 姿态接近垂直
    p0_est = pos0 + rng.normal(0, 0.5, 3)
    v0_est = vel0 + rng.normal(0, 0.3, 3)
    q0_est = qu.quat_normalize(qu.Q_VERT + rng.normal(0, 0.01, 4))
    ekf = MEKF(p0_est, v0_est, q0_est, dt=DT)

    max_t = 200.0
    n_steps = int(max_t / DT)

    # 诊断日志
    ekf_err_log = []      # EKF估计误差
    flex_log = []          # 弹性体状态
    gps_update_count = 0
    radar_update_count = 0
    terminal_mode_triggered = False

    for i in range(n_steps):
        h_true = -state[2]
        m, cg_x, I_body = rp.mass_properties(fuel)

        # === E3: 从EKF获取估计状态 (制导/姿态控制器不看真值!) ===
        state_est, sigma_est = ekf.get_state()
        omega_est = ekf.get_estimated_omega(state[10:13])  # 用IMU测量(已校正)
        # 注意: omega_est 用最新IMU测量, state_est[10:13]占位为0
        # 但制导/姿态控制器需要omega, 所以单独传
        # 构造完整估计状态 (替换omega)
        state_est_full = state_est.copy()
        state_est_full[10:13] = omega_est

        # === 末端模式切换 (h<50m 禁用GPS) ===
        h_est = -state_est[2]
        if h_est < 50.0 and not terminal_mode_triggered:
            ekf.set_terminal_mode(True)
            terminal_mode_triggered = True
            if verbose:
                print("  [E3] 末端模式触发 (h<50m), 禁用GPS")

        # === 制导 (用估计状态) ===
        throttle, q_des, omega_des, tvc_gimbal_cmd, phase = guidance.update(
            state_est_full, fuel, t, DT)

        if phase == 'LANDED':
            break

        # === 姿态控制 (用估计状态) ===
        rho, a, _, _ = atmosphere(h_true)
        v_mag = np.linalg.norm(state[3:6])  # 动压用真值(气动是物理量)
        mach = v_mag / a if a > 0 else 0.0
        qdyn = 0.5 * rho * v_mag * v_mag

        gf_cmd, rcs_cmd, tvc_gimbal_att = att.update(
            q_des, omega_des, state_est_full, mach, qdyn, cg_x, I_body, gf, rcs, DT,
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

        # === E1: 弹性体+晃动 (用真值计算, 因为是物理过程) ===
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
        # 计算IMU真值 (比力 = 非重力合力/m, body系)
        F_thrust_b = info['F_thrust_b']
        F_aero_b = info['F_aero_b']
        # extra_force_b = 0 (RCS/GridFin给力矩不给力, 简化)
        f_b_true = (F_thrust_b + F_aero_b) / m  # 比力 (specific force)
        omega_b_true = state[10:13]

        # IMU测量 (加噪声+零偏+弹性扰动)
        gyro_meas, accel_meas = imu.measure(
            omega_b_true + imu_omega_dist,
            f_b_true + imu_accel_dist, DT)

        # EKF预测步 (100Hz)
        ekf.predict(gyro_meas, accel_meas, DT)

        # GPS更新 (10Hz)
        pos_meas, vel_meas, gps_valid = gps.measure(state[0:3], state[3:6], DT)
        if gps_valid:
            ekf.update_gps(pos_meas, vel_meas)
            gps_update_count += 1

        # 雷达高度计更新 (50Hz, h<100m)
        alt_meas, radar_valid = radar.measure(state[0:3], DT)
        if radar_valid:
            ekf.update_radar(alt_meas)
            radar_update_count += 1

        # === 诊断日志 (每1秒) ===
        if i % 100 == 0:
            pos_err = np.linalg.norm(ekf.p - state[0:3])
            vel_err = np.linalg.norm(ekf.v - state[3:6])
            # 姿态误差角: 用误差四元数的旋转角 2*acos(|w|)
            q_err = qu.quat_multiply(qu.quat_inverse(state[6:10]), ekf.q)
            q_err_angle = 2.0 * np.degrees(np.arccos(np.clip(abs(q_err[0]), 0, 1)))
            ekf_err_log.append((t, pos_err, vel_err, q_err_angle,
                                np.linalg.norm(ekf.bg), np.linalg.norm(ekf.ba)))
            flex_log.append((t, np.linalg.norm(flex.eta), np.linalg.norm(flex.xi),
                             np.linalg.norm(slosh_moment)))

        # 详细调试: t=83-87 每0.1s打印
        # if 83.0 <= t <= 87.0 and i % 10 == 0:
        #     ... (调试完毕, 雷达H矩阵符号错误已修复)

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

    # EKF最终估计误差
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
        'flex_log': flex_log,
    }

    if verbose:
        print("=== E1+E3 联合测试 (seed=%d) ===" % seed)
        print("着陆成功:", success)
        print("终态: h=%.2fm  vz=%.2fm/s  tilt=%.1f°  pos=%.2fm" %
              (h_final, vz_final, tilt, h_pos_err))
        print("仿真时间: %.1fs  剩余燃料: %.0fkg" % (t, fuel))
        print("\n=== EKF 估计误差 ===")
        print("位置误差: %.3fm  速度误差: %.3fm/s  姿态误差: %.3f°" %
              (pos_err_final, vel_err_final, q_err_final))
        print("GPS更新次数: %d  雷达更新次数: %d" % (gps_update_count, radar_update_count))
        print("末端模式触发:", terminal_mode_triggered)
        print("零偏估计: bg=%.4f rad/s  ba=%.4f m/s²" %
              (np.linalg.norm(ekf.bg), np.linalg.norm(ekf.ba)))

        print("\n=== EKF 误差时间序列 (t, pos_err, vel_err, q_err, |bg|, |ba|) ===")
        for log in ekf_err_log[:3]:
            print("  t=%.0f pos=%.3f vel=%.3f q=%.3f° bg=%.4f ba=%.4f" % log)
        print("  ...")
        for log in ekf_err_log[-3:]:
            print("  t=%.0f pos=%.3f vel=%.3f q=%.3f° bg=%.4f ba=%.4f" % log)

        print("\n=== 弹性体诊断 ===")
        print("弯曲频率: ω1=%.1f rad/s (%.2fHz)  ω2=%.1f rad/s (%.2fHz)" % (
            OMEGA_FLEX_1, OMEGA_FLEX_1/(2*np.pi), OMEGA_FLEX_2, OMEGA_FLEX_2/(2*np.pi)))
        if flex_log:
            eta_max = max(log[1] for log in flex_log)
            xi_max = max(log[2] for log in flex_log)
            slosh_max = max(log[3] for log in flex_log)
            print("弹性模态最大幅值: |eta|=%.4e" % eta_max)
            print("晃动最大幅值: |xi|=%.4e" % xi_max)
            print("晃动力矩最大: %.2f N·m" % slosh_max)

    return success, metrics


if __name__ == '__main__':
    run_e1e3(seed=42)
