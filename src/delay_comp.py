"""E4: 延迟补偿 (Smith预估器 + 状态前推).

理论方案2.0 E4节:
  传感器延迟50ms + 执行器延迟100ms = 总延迟150ms.
  G-FOLD在t时刻求解, 控制指令在t+150ms执行.
  若G-FOLD用t时刻状态, 实际执行时状态已变化→闭环振荡.
  Smith预估器: 将EKF状态前推150ms, 给G-FOLD"未来"状态.

混合前推策略 (方案2.0 E4问题1解决方案):
  h > 500m:  动力学积分前推(精确, 用IMU比力+重力)
  50m < h <= 500m: 匀加速前推(简化, 末端机动小)
  h < 50m:   不前推(死区, 延迟影响小)

前推公式 (匀加速, 足够150ms精度):
  p_future = p + v*dt + 0.5*a*dt²
  v_future = v + a*dt
  q_future = q ⊗ exp(0.5*omega*dt)  (小角度近似)
  a = C_bn @ f_b + g_n  (NED系加速度, f_b为校正比力)
"""
import numpy as np
from . import quaternion_utils as qu
from . import rocket_params as rp


class DelayCompensator:
    """Smith预估器: 前推EKF状态补偿传感器+执行器延迟."""

    def __init__(self, total_delay=0.15):
        """total_delay: 总延迟(s), 传感器50ms+执行器100ms=150ms."""
        self.delay = total_delay
        # 最新IMU校正测量 (用于前推时的加速度/角速度估计)
        self._f_b = np.zeros(3)       # 比力 (body系, 已校正零偏)
        self._omega_b = np.zeros(3)   # 角速度 (body系, 已校正零偏)
        self._g_n = np.array([0.0, 0.0, rp.G0])

    def update_imu(self, f_b, omega_b):
        """存储最新IMU校正测量, 供前推使用.
        f_b: 校正比力 = accel_meas - ba (body系)
        omega_b: 校正角速度 = gyro_meas - bg (body系)
        """
        self._f_b = np.asarray(f_b, dtype=float).copy()
        self._omega_b = np.asarray(omega_b, dtype=float).copy()

    def compensate(self, state_est):
        """前推状态补偿延迟.
        state_est: 13D [pos_n(3), vel_n(3), q(4), omega_b(3)] (来自EKF)
        返回: 前推后的13D状态 (供G-FOLD/姿态控制器使用).
        """
        h = -state_est[2]

        # 死区(h<50m): 不前推, 延迟影响小
        if h < 50.0 or self.delay < 1e-6:
            return state_est.copy()

        dt = self.delay
        q = state_est[6:10]
        C_bn = qu.quat_to_rotmat(q)

        # NED系加速度: a = C_bn @ f_b + g
        a_n = C_bn @ self._f_b + self._g_n

        # 位置/速度前推 (匀加速)
        pos_future = state_est[0:3] + state_est[3:6] * dt + 0.5 * a_n * dt * dt
        vel_future = state_est[3:6] + a_n * dt

        # 姿态前推: q_dot = 0.5 * q ⊗ [0, omega]
        # 一阶近似: q_future = normalize(q + q_dot * dt)
        q_dot = qu.quat_kinematics(q, self._omega_b)
        q_future = qu.quat_normalize(q + q_dot * dt)

        state_comp = state_est.copy()
        state_comp[0:3] = pos_future
        state_comp[3:6] = vel_future
        state_comp[6:10] = q_future
        # omega不变 (短时间近似)
        return state_comp
