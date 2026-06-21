"""Step D: 无损凸化验证. 跑T_MIN=0.40和T_MIN=0.15两个配置."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import src.quaternion_utils as qu
import src.rocket_params as rp
import src.guidance as g
from src.dynamics import DynamicsEngine, make_state
from src.actuators import TVC, GridFin, RCS
from src.attitude_control import AttitudeController
from src.guidance import LandingGuidance
from src.wind import DrydenWind
from src.atmosphere import atmosphere

DT = 0.01


def run_single_landing(pos0, vel0, fuel_init, rng_seed, wind_sigma=1.0, t_min_frac=None):
    """跑一次着陆, 可选修改T_MIN_FRAC."""
    if t_min_frac is not None:
        g.T_MIN_FRAC = t_min_frac

    rng = np.random.default_rng(rng_seed)
    dyn = DynamicsEngine(dt=DT)
    tvc = TVC(tau=0.05)
    gf = GridFin()
    rcs = RCS()
    att = AttitudeController(wn=2 * np.pi * 0.3, zeta=0.9)
    guidance = LandingGuidance(gfold_N=50, dt=DT)
    guidance.gfold._step_d_debug = True  # 开启Step D打印
    wind = DrydenWind(rng, sigma_base=wind_sigma)

    state = make_state(pos_n=pos0, vel_n=vel0, q=qu.Q_VERT.copy())
    fuel = fuel_init
    t = 0.0
    M_aero_prev = np.zeros(3)
    max_t = 200.0
    n_steps = int(max_t / DT)
    first_solve_printed = False

    for i in range(n_steps):
        h = -state[2]
        m, cg_x, I_body = rp.mass_properties(fuel)

        throttle, q_des, omega_des, tvc_gimbal_cmd, phase = guidance.update(
            state, fuel, t, DT)

        if phase == 'LANDED':
            break

        rho, a, _, _ = atmosphere(h)
        v_mag = np.linalg.norm(state[3:6])
        mach = v_mag / a if a > 0 else 0.0
        qdyn = 0.5 * rho * v_mag * v_mag

        gf_cmd, rcs_cmd, tvc_gimbal_att = att.update(
            q_des, omega_des, state, mach, qdyn, cg_x, I_body, gf, rcs, DT,
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
            throttle, tvc_pitch_cmd, tvc_yaw_cmd, phase, h, DT)

        w = wind.update(h, state[3:6], DT)

        state, fuel, info = dyn.step(
            state, fuel, thrust_actual, gp, gy, w,
            extra_moment_b=M_gf + M_rcs)
        M_aero_prev = info['M_aero_b'].copy()

        # Step C: 提取气动力转NED系
        F_aero_b = info['F_aero_b']
        C_bn = qu.quat_to_rotmat(state[6:10])
        guidance.F_aero_n_prev = C_bn @ F_aero_b

        t += DT

        if h < 0.0 and abs(state[5]) > 15.0:
            break

    h_final = -state[2]
    vz_final = state[5]
    tilt = np.degrees(qu.tilt_angle_from_vertical(state[6:10]))
    h_pos_err = np.hypot(state[0], state[1])

    success = (h_final < 1.0 and abs(vz_final) < 5.0 and
               tilt < 45.0 and h_pos_err < 50.0)

    return success, {
        'h_final': h_final, 'vz_final': vz_final, 'tilt': tilt,
        'h_pos_err': h_pos_err, 't': t, 'fuel': fuel
    }


if __name__ == '__main__':
    out = []
    pos0 = np.array([0.0, 0.0, -2000.0])
    vel0 = np.array([0.0, 0.0, 80.0])

    # 配置1: T_MIN=0.40
    print("=" * 60)
    print("配置1: T_MIN_FRAC=0.40 (T_min=338kN > 重力275kN)")
    print("=" * 60)
    success, info = run_single_landing(pos0, vel0, 15000.0, 42, wind_sigma=1.0, t_min_frac=0.40)
    line1 = "T_MIN=0.40: success=%s h=%.2f vz=%.2f fuel=%.0f t=%.1f" % (
        success, info['h_final'], info['vz_final'], info['fuel'], info['t'])
    out.append(line1)
    print(line1)

    # 配置2: T_MIN=0.15
    print("\n" + "=" * 60)
    print("配置2: T_MIN_FRAC=0.15 (T_min=127kN < 重力275kN, 可悬停)")
    print("=" * 60)
    success, info = run_single_landing(pos0, vel0, 15000.0, 42, wind_sigma=1.0, t_min_frac=0.15)
    line2 = "T_MIN=0.15: success=%s h=%.2f vz=%.2f fuel=%.0f t=%.1f" % (
        success, info['h_final'], info['vz_final'], info['fuel'], info['t'])
    out.append(line2)
    print(line2)

    with open('step_d_result.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print("\n完成")
