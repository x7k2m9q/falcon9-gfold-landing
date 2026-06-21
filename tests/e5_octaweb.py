"""E5: 多发动机1-3-1剖面集成测试 (Octaweb + LandingProfile).

在E1+E3+E4基础上加入E5多发动机模型:
  - Octaweb: 9台Merlin 1D (8外围+1中心), 1-3-1点火策略
  - LandingProfile: 3发(vz>40或tgo>15s) / 1发(vz<=40且tgo<=15s)
  - G-FOLD SOCP: T_min/T_max随发动机数缩放, 终端约束自适应

物理分析:
  3发模式: T_min=3*338=1014kN >> 重力363kN → 无法悬停, 仅用于高速制动
    - vz=80→40: 净上推力651kN, 减速度17.6m/s², 制动时间2.3s, 距离137m
  1发模式: T_min=338kN < 重力363kN → hover-slam精确着陆
    - vz=40→3: bang-bang/PD控制, 持续减速至着陆

验证项:
  1. 1-3-1切换正确 (3发制动→1发着陆)
  2. 着陆精度不降级 (对比E4)
  3. 发动机动态 (点火/关机/抖动) 不影响稳定性
  4. EKF估计误差在合理范围
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
from src.octaweb import Octaweb, LandingProfile

DT = 0.01


def run_e5(seed=42, verbose=True):
    """运行E5多发动机仿真. 返回 (success, metrics_dict)."""
    rng = np.random.default_rng(seed)
    dyn = DynamicsEngine(dt=DT)
    # E5: 用Octaweb替代单TVC
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
    engine_config_log = []  # E5: 记录发动机配置变化
    gps_update_count = 0
    radar_update_count = 0
    terminal_mode_triggered = False

    prev_gyro_meas = np.zeros(3)
    prev_accel_meas = np.zeros(3)

    # E5: 初始1发 (DESCENT段), G-FOLD段由LandingProfile决策
    octaweb.set_engine_config(1)
    prev_n_engines = 1

    for i in range(n_steps):
        h_true = -state[2]
        m, cg_x, I_body = rp.mass_properties(fuel)

        # === E3: EKF状态估计 ===
        state_est, sigma_est = ekf.get_state()
        omega_est = ekf.get_estimated_omega(prev_gyro_meas)
        state_est_full = state_est.copy()
        state_est_full[10:13] = omega_est

        # === E4: 延迟补偿 ===
        f_b_prev = prev_accel_meas - ekf.ba
        omega_b_prev = prev_gyro_meas - ekf.bg
        delay_comp.update_imu(f_b_prev, omega_b_prev)
        state_comp = delay_comp.compensate(state_est_full)

        # === 末端模式 ===
        h_est = -state_est[2]
        if h_est < 50.0 and not terminal_mode_triggered:
            ekf.set_terminal_mode(True)
            terminal_mode_triggered = True

        # === 制导 ===
        throttle, q_des, omega_des, tvc_gimbal_cmd, phase = guidance.update(
            state_comp, fuel, t, DT)

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

        # === 姿态控制 ===
        rho, a, _, _ = atmosphere(h_true)
        v_mag = np.linalg.norm(state[3:6])
        mach = v_mag / a if a > 0 else 0.0
        qdyn = 0.5 * rho * v_mag * v_mag

        gf_cmd, rcs_cmd, tvc_gimbal_att = att.update(
            q_des, omega_des, state_comp, mach, qdyn, cg_x, I_body, gf, rcs, DT,
            M_disturbance=M_aero_prev, phase=phase, tvc=octaweb.center_tvc,
            thrust_actual=octaweb.center_tvc.thrust)

        M_gf = gf.update(gf_cmd, mach, qdyn, cg_x, DT)
        M_rcs = rcs.update(rcs_cmd, cg_x, DT)

        if phase in ('G-FOLD', 'DEADBAND'):
            tvc_pitch_cmd = tvc_gimbal_att[0]
            tvc_yaw_cmd = tvc_gimbal_att[1]
        else:
            tvc_pitch_cmd = 0.0
            tvc_yaw_cmd = 0.0

        # === E5: Octaweb更新 (替代单TVC) ===
        total_thrust, gp, gy = octaweb.update(
            throttle, tvc_pitch_cmd, tvc_yaw_cmd,
            phase, h_true, DT, rng=rng)

        w = wind.update(h_true, state[3:6], DT)

        # === E1: 弹性体+晃动 ===
        C_bn = qu.quat_to_rotmat(state[6:10])
        a_thrust_b = np.array([total_thrust * np.cos(gp) * np.cos(gy),
                               total_thrust * np.sin(gp),
                               total_thrust * np.cos(gp) * np.sin(gy)]) / m
        a_grav_n = np.array([0, 0, rp.G0])
        a_grav_b = C_bn.T @ a_grav_n
        a_lateral_b = np.array([a_thrust_b[1] + a_grav_b[1],
                                a_thrust_b[2] + a_grav_b[2]])
        omega_b_true = state[10:13]
        thrust_b = np.array([total_thrust * np.cos(gp) * np.cos(gy),
                             total_thrust * np.sin(gp),
                             total_thrust * np.cos(gp) * np.sin(gy)])
        imu_omega_dist, imu_accel_dist, aero_moment_dist, slosh_moment = \
            flex.update(DT, a_lateral_b, omega_b_true, fuel, thrust_b)

        # E5: 外围发动机偏心力矩 (对称配置下近似为0, 但噪声残留)
        F_octa, M_octa = octaweb.compute_force_moment(gp, gy, cg_x)
        # 中心发动机力矩已由dynamics.py计算, 只加外围偏心力矩
        # M_octa包含所有发动机力矩, 减去中心发动机力矩
        x_tvc = -rp.LENGTH / 2.0 + 1.0
        r_center = np.array([x_tvc - cg_x, 0.0, 0.0])
        F_center = np.array([octaweb.engines[0].thrust * np.cos(gp) * np.cos(gy),
                             octaweb.engines[0].thrust * np.sin(gp),
                             octaweb.engines[0].thrust * np.cos(gp) * np.sin(gy)])
        M_center = np.cross(r_center, F_center)
        M_outer_residual = M_octa - M_center

        extra_moment = M_gf + M_rcs + slosh_moment + aero_moment_dist + M_outer_residual

        # === 动力学推进 ===
        # E5: 用总推力和中心TVC角度. 对称配置下外围力矩抵消.
        state, fuel, info = dyn.step(
            state, fuel, total_thrust, gp, gy, w,
            extra_moment_b=extra_moment)
        M_aero_prev = info['M_aero_b'].copy()

        # === E3: 传感器 + EKF更新 ===
        F_thrust_b = info['F_thrust_b']
        F_aero_b = info['F_aero_b']
        f_b_true = (F_thrust_b + F_aero_b) / m
        omega_b_true = state[10:13]

        gyro_meas, accel_meas = imu.measure(
            omega_b_true + imu_omega_dist,
            f_b_true + imu_accel_dist, DT)

        prev_gyro_meas = gyro_meas.copy()
        prev_accel_meas = accel_meas.copy()

        ekf.predict(gyro_meas, accel_meas, DT)

        pos_meas, vel_meas, gps_valid = gps.measure(state[0:3], state[3:6], DT)
        if gps_valid:
            ekf.update_gps(pos_meas, vel_meas)
            gps_update_count += 1

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
        'engine_config_log': engine_config_log,
        'final_n_engines': prev_n_engines,
    }

    if verbose:
        print("=== E5 多发动机1-3-1 (seed=%d) ===" % seed)
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

    return success, metrics


if __name__ == '__main__':
    print("=" * 60)
    print("E5: 多发动机1-3-1剖面集成测试")
    print("=" * 60)

    s, m = run_e5(seed=42, verbose=True)

    print("\n" + "=" * 60)
    print("总结:")
    print("  着陆: %s  h=%.2fm  vz=%.2fm/s  tilt=%.1f°" %
          (m['success'], m['h_final'], m['vz_final'], m['tilt_final']))
    print("  燃料: %.0fkg  时间: %.1fs" % (m['fuel_final'], m['t_final']))
    print("  EKF: pos=%.3fm vel=%.3fm/s q=%.3f°" %
          (m['pos_err_ekf'], m['vel_err_ekf'], m['q_err_ekf']))
