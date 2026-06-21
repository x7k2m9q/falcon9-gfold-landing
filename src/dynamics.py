"""
6DOF 刚体动力学引擎 (纯 NumPy, RK4).
状态 x[13] = [pos_n(3), vel_n(3), q(4), omega_b(3)]
  pos_n: NED 位置 [北, 东, 地下], 高度 h = -pos_n[2]
  vel_n: NED 速度
  q: b->n 四元数 [w,x,y,z]
  omega_b: b系角速度 [wx,wy,wz]
力/力矩:
  推力沿 +Xb, TVC偏转: thrust_dir_b = [cos(θz)cos(θy), sin(θz)cos(θy), -sin(θy)]
    θy=gimbal_pitch(绕Yb), θz=gimbal_yaw(绕Zb)
  推力作用点 r_tvc=[x_tvc-cg_x,0,0] (尾部), M_thrust = r_tvc × F_thrust_b
  气动力/矩由 aero 模块在 b系 给出
  外部力矩 extra_moment_b (RCS/栅格舵, Step2 注入)
  重力 [0,0,m*g] 在 n系
积分: RK4, 每子步重算 mass_properties 与气动 (随状态变化). 步末扣燃料.
"""
import numpy as np
from . import quaternion_utils as qu
from . import rocket_params as rp
from . import aero as aero_mod


def make_state(pos_n=None, vel_n=None, q=None, omega_b=None):
    s = np.zeros(13)
    s[0:3] = pos_n if pos_n is not None else np.zeros(3)
    s[3:6] = vel_n if vel_n is not None else np.zeros(3)
    s[6:10] = q if q is not None else qu.Q_VERT.copy()
    s[10:13] = omega_b if omega_b is not None else np.zeros(3)
    return s


def get_pos(s): return s[0:3]
def get_vel(s): return s[3:6]
def get_quat(s): return s[6:10]
def get_omega(s): return s[10:13]


def thrust_dir_b(gimbal_pitch, gimbal_yaw):
    """推力方向单位向量(b系). gimbal_pitch=θy(绕Yb), gimbal_yaw=θz(绕Zb)."""
    cp = np.cos(gimbal_pitch)
    return np.array([cp * np.cos(gimbal_yaw),
                     cp * np.sin(gimbal_yaw),
                     -np.sin(gimbal_pitch)])


def _derivs(state, fuel_mass, thrust_N, g_pitch, g_yaw,
            wind_n, extra_force_b, extra_moment_b):
    """计算 dx/dt. 返回 (dxdt[13], info_dict)."""
    pos = state[0:3]
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    h = -pos[2]
    m, cg_x, I_body = rp.mass_properties(fuel_mass)
    C_bn = qu.quat_to_rotmat(q)
    # 推力
    tdir = thrust_dir_b(g_pitch, g_yaw)
    F_thrust_b = thrust_N * tdir
    F_thrust_n = C_bn @ F_thrust_b
    # 推力力矩 (作用点尾部)
    x_tvc = -rp.LENGTH / 2.0 + 1.0
    r_tvc = np.array([x_tvc - cg_x, 0.0, 0.0])
    M_thrust_b = np.cross(r_tvc, F_thrust_b)
    # 气动
    F_aero_b, M_aero_b, alpha, mach, qdyn = aero_mod.aero_forces_and_moments(
        vel, wind_n, q, h, cg_x)
    # 总力 (n系)
    F_extra_n = C_bn @ extra_force_b
    F_grav_n = np.array([0.0, 0.0, m * rp.G0])
    F_total_n = F_thrust_n + C_bn @ F_aero_b + F_extra_n + F_grav_n
    # 总力矩 (b系)
    M_total_b = M_thrust_b + M_aero_b + extra_moment_b
    # 导数
    pos_dot = vel
    vel_dot = F_total_n / m
    q_dot = qu.quat_kinematics(q, omega)
    Iw = I_body @ omega
    omega_dot = np.linalg.solve(I_body, M_total_b - np.cross(omega, Iw))
    dxdt = np.concatenate([pos_dot, vel_dot, q_dot, omega_dot])
    info = dict(alpha=alpha, mach=mach, qdyn=qdyn, mass=m, cg_x=cg_x,
                F_thrust_b=F_thrust_b, M_thrust_b=M_thrust_b,
                F_aero_b=F_aero_b, M_aero_b=M_aero_b, h=h)
    return dxdt, info


class DynamicsEngine:
    def __init__(self, dt=0.01):
        self.dt = dt

    def step(self, state, fuel_mass, thrust_N, gimbal_pitch, gimbal_yaw,
             wind_n, extra_force_b=None, extra_moment_b=None):
        """RK4 推进一步. 返回 (new_state, new_fuel, info)."""
        if extra_force_b is None:
            extra_force_b = np.zeros(3)
        if extra_moment_b is None:
            extra_moment_b = np.zeros(3)
        dt = self.dt

        def f(s):
            d, _ = _derivs(s, fuel_mass, thrust_N, gimbal_pitch, gimbal_yaw,
                           wind_n, extra_force_b, extra_moment_b)
            return d

        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # 四元数归一化
        new_state[6:10] = qu.quat_normalize(new_state[6:10])
        # 燃料消耗 (用步初推力估计)
        h0 = -state[2]
        mdot = rp.fuel_flow_rate(thrust_N, h0)
        new_fuel = max(0.0, fuel_mass - mdot * dt)
        # 末态 info
        _, info = _derivs(new_state, new_fuel, thrust_N, gimbal_pitch, gimbal_yaw,
                          wind_n, extra_force_b, extra_moment_b)
        return new_state, new_fuel, info
