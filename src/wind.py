"""
Dryden 大气湍流模型 (一阶有色噪声滤波, 严禁白噪声).
公式: V(t) = V(t-dt)*exp(-dt/tau) + sigma*sqrt(1-exp(-2*dt/tau))*n(t)
tau, sigma 随高度变化. 低空湍流强, 高空弱.
三维独立通道 (u纵向北, v东, w垂直). 每步用当前火箭速度估计时间常数.
"""
import numpy as np
from . import atmosphere


class DrydenWind:
    def __init__(self, rng,
                 sigma_base=1.8,        # 地表湍流强度 m/s (轻-中度)
                 L_base=200.0,          # 低空湍流尺度 m
                 h_ref=300.0,           # 参考高度
                 seed_wind=None):
        self.rng = rng
        self.sigma_base = sigma_base
        self.L_base = L_base
        self.h_ref = h_ref
        # 初始风速 (小量, 模拟背景风)
        self.wind_n = np.zeros(3)
        if seed_wind is not None:
            self.wind_n = np.array(seed_wind, dtype=float)

    def _params(self, h, v_mag):
        """高度h, 火箭对地速度v_mag -> (sigma_vec[3], tau_vec[3])."""
        # 湍流强度随高度: 低空强(地表对流), 高空指数衰减
        # h<300m 近似恒定, 之后衰减
        if h < self.h_ref:
            factor = 0.6 + 0.4 * (h / self.h_ref)  # 地表稍弱, 300m最强
        else:
            factor = np.exp(-(h - self.h_ref) / 3000.0) + 0.15
        sigma = self.sigma_base * factor
        # 三轴湍流强度: 垂直分量(w)通常小于水平
        sigma_vec = np.array([sigma, sigma * 0.9, sigma * 0.7])
        # 尺度长度: 低空L_base, 高空随高度增大
        if h < self.h_ref:
            L = self.L_base
        else:
            L = self.L_base + (h - self.h_ref) * 0.5
        # 时间常数 tau = L / V (V最小取10避免tau爆炸)
        v_eff = max(v_mag, 10.0)
        tau_vec = np.array([L / v_eff, L / v_eff, L / (v_eff * 1.5)])
        return sigma_vec, tau_vec

    def update(self, h, v_ground_n, dt):
        """推进风场一步. v_ground_n: 火箭对地速度(n系). 返回当前风速(n系)."""
        v_mag = np.linalg.norm(v_ground_n)
        sigma_vec, tau_vec = self._params(max(h, 0.0), v_mag)
        decay = np.exp(-dt / tau_vec)
        noise_gain = sigma_vec * np.sqrt(1.0 - decay ** 2)
        n = self.rng.normal(0.0, 1.0, size=3)
        self.wind_n = self.wind_n * decay + noise_gain * n
        return self.wind_n.copy()

    def get(self):
        return self.wind_n.copy()
