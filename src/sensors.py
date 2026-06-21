"""
传感器模型: 陀螺/加计/GPS. 第一天内置噪声, 不允许无噪版本.
高斯白噪声 + 零偏漂移 (随机游走).
"""
import numpy as np


class IMU:
    """陀螺+加计. 输出带噪声和零偏的测量值."""

    def __init__(self, rng,
                 gyro_noise_std=np.radians(0.01),     # rad/s/sqrt(Hz) 简化
                 accel_noise_std=0.02,                # m/s^2/sqrt(Hz)
                 gyro_bias_walk=np.radians(1e-4),     # rad/s 零偏游走/步
                 accel_bias_walk=1e-4,                # m/s^2 零偏游走/步
                 gyro_init_bias=np.radians(0.5),      # 初始零偏量级
                 accel_init_bias=0.05):
        self.rng = rng
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gyro_bias_walk = gyro_bias_walk
        self.accel_bias_walk = accel_bias_walk
        # 初始零偏 (随机)
        self.gyro_bias = rng.normal(0.0, gyro_init_bias, size=3)
        self.accel_bias = rng.normal(0.0, accel_init_bias, size=3)

    def measure(self, omega_body_true, accel_body_true, dt):
        """omega_body_true: b系真实角速度. accel_body_true: b系真实比力(不含重力)."""
        # 零偏随机游走
        self.gyro_bias += self.rng.normal(0.0, self.gyro_bias_walk * np.sqrt(dt), size=3)
        self.accel_bias += self.rng.normal(0.0, self.accel_bias_walk * np.sqrt(dt), size=3)
        gyro_meas = omega_body_true + self.gyro_bias + self.rng.normal(0.0, self.gyro_noise_std, size=3)
        accel_meas = accel_body_true + self.accel_bias + self.rng.normal(0.0, self.accel_noise_std, size=3)
        return gyro_meas, accel_meas


class GPS:
    """GPS: 位置+速度测量, 低频(10Hz), 带噪声."""

    def __init__(self, rng,
                 pos_noise_std=0.5,    # m
                 vel_noise_std=0.1,    # m/s
                 update_rate=10.0):    # Hz
        self.rng = rng
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.update_rate = update_rate
        self._acc = 0.0

    def measure(self, pos_true, vel_true, dt):
        """返回 (pos_meas, vel_meas, valid). valid=True 表示本步有更新."""
        self._acc += dt
        if self._acc >= 1.0 / self.update_rate:
            self._acc = 0.0
            pos_meas = pos_true + self.rng.normal(0.0, self.pos_noise_std, size=3)
            vel_meas = vel_true + self.rng.normal(0.0, self.vel_noise_std, size=3)
            return pos_meas, vel_meas, True
        return None, None, False


class RadarAltimeter:
    """雷达高度计: 高精度高度测量, 末端50m以下使用.
    高频(50Hz), 噪声σ=0.05m, 无累积漂移."""

    def __init__(self, rng,
                 alt_noise_std=0.05,   # m
                 update_rate=50.0,     # Hz
                 max_alt=100.0):       # m, 有效量程
        self.rng = rng
        self.alt_noise_std = alt_noise_std
        self.update_rate = update_rate
        self.max_alt = max_alt
        self._acc = 0.0

    def measure(self, pos_true, dt):
        """返回 (alt_meas, valid). alt_meas = -pos_n[2] (高度, 向上为正).
        valid=True表示本步有更新且在量程内."""
        self._acc += dt
        h_true = -pos_true[2]
        if self._acc >= 1.0 / self.update_rate and h_true < self.max_alt:
            self._acc = 0.0
            # 末端近距离时噪声略增(地表多径)
            noise_scale = 1.0 + max(0.0, (10.0 - h_true) / 10.0) * 0.5
            alt_meas = h_true + self.rng.normal(0.0, self.alt_noise_std * noise_scale)
            return alt_meas, True
        return None, False
