"""
气动模型: 攻角/侧滑角, 阻力/升力系数, 跨音速马赫数修正.
力与力矩在 b系 计算.
相对速度 V_rel = V_body_n - V_wind_n, 经 C_n^b 转到 b系.
"""
import numpy as np
from . import atmosphere, rocket_params


def mach_correction(M):
    """跨音速压缩因子 (乘到Cd上). 亚音速Prandtl-Glauert, 跨音速峰, 超音速衰减."""
    if M < 0.7:
        return 1.0 / np.sqrt(max(1e-6, 1.0 - M * M))
    elif M < 1.0:
        # 0.7->1.0 平滑过渡到峰
        k = (M - 0.7) / 0.3
        return (1.0 / np.sqrt(1.0 - 0.49)) * (1.0 - k) + 2.5 * k
    elif M < 1.2:
        # 跨音速峰 1.0-1.2
        k = (M - 1.0) / 0.2
        return 2.5 * (1.0 - k) + 1.8 * k
    else:
        # 超音速衰减 ~ 1/M^0.5
        return max(1.0, 1.8 * np.sqrt(1.2 / M))


def aero_forces_and_moments(v_body_n, wind_n, q, h, cg_x):
    """计算气动力(b系)与力矩(b系).
    返回 (F_aero_b[3], M_aero_b[3], alpha_tot, mach, dyn_pressure).
    """
    rho, a_sound, _, _ = atmosphere.atmosphere(max(h, 0.0))
    # 相对速度 n系 -> b系
    v_rel_n = v_body_n - wind_n
    from .quaternion_utils import quat_to_rotmat
    C_bn = quat_to_rotmat(q)          # b->n
    C_nb = C_bn.T                     # n->b
    v_rel_b = C_nb @ v_rel_n
    v_mag = np.linalg.norm(v_rel_b)
    if v_mag < 0.5 or rho < 1e-6:
        return np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0
    mach = v_mag / a_sound
    # 攻角/侧滑
    vx, vy, vz = v_rel_b
    alpha = np.arctan2(vz, vx)        # 俯仰面攻角 (Xb-Zb)
    beta = np.arcsin(np.clip(vy / v_mag, -1.0, 1.0))  # 侧滑 (Xb-Yb)
    alpha_tot = np.arctan2(np.hypot(vy, vz), vx)
    # 系数
    Cd = 0.25 + 0.5 * np.sin(alpha_tot) ** 2
    Cl = 0.8 * np.sin(2.0 * alpha_tot)
    Cd *= mach_correction(mach)
    q_dyn = 0.5 * rho * v_mag * v_mag
    S = rocket_params.REF_AREA
    # 阻力沿 -V_rel 方向
    F_drag_b = -q_dyn * S * Cd * (v_rel_b / v_mag)
    # 升力: 垂直 V_rel, 在 (V_rel, Xb) 平面, 指向减小攻角
    Xb = np.array([1.0, 0.0, 0.0])
    v_hat = v_rel_b / v_mag
    e_lift = Xb - np.dot(Xb, v_hat) * v_hat
    n_lift = np.linalg.norm(e_lift)
    if n_lift > 1e-9:
        e_lift = e_lift / n_lift
        F_lift_b = q_dyn * S * Cl * e_lift
    else:
        F_lift_b = np.zeros(3)
    F_aero_b = F_drag_b + F_lift_b
    # 气动力矩: 压力中心相对质心. 静稳定: 压心在质心后方(-Xb).
    # 压心位置 x_cp (相对几何中心), 转相对质心: x_cp - cg_x
    x_cp = -3.0  # 压心在尾部方向 (相对几何中心)
    r_cp = np.array([x_cp - cg_x, 0.0, 0.0])
    M_aero_b = np.cross(r_cp, F_aero_b)
    return F_aero_b, M_aero_b, alpha_tot, mach, q_dyn
