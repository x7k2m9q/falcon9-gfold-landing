"""E5: 多发动机1-3-1剖面 (Octaweb + 控制分配).

理论方案2.0 E5节:
  9台Merlin 1D发动机 (8外围+1中心).
  1-3-1点火策略:
    3发模式: 中心+2外围, 暴力减速 (tgo>15s 或 vz>40)
    1发模式: 仅中心, 精确调节 (tgo<=15s 且 vz<=40)

物理模型 (Falcon 9实际设计):
  中心发动机: 双向TVC (pitch+yaw gimbal), 提供姿态控制力矩
  外围8台: 固定推力方向(无TVC), 仅提供轴向推力
  每台发动机独立动态(E2): 点火/关机/抖动/延迟

控制分配 (简化QP, 因仅中心有TVC):
  轴向推力: G-FOLD总推力 → 均分到各活跃发动机
  姿态力矩: 姿态控制器期望力矩 → 中心发动机TVC偏转
  外围发动机不参与姿态控制, 仅提供推力

1-3-1切换:
  3发→1发: 关闭2台外围发动机(关机尾推0.8s衰减)
  G-FOLD的T_min/T_max随发动机数变化:
    3发: T_min=3*338kN=1014kN, T_max=3*845kN=2535kN
    1发: T_min=338kN, T_max=845kN
"""
import numpy as np
from . import rocket_params as rp
from .actuators import TVC


class EngineDynamics:
    """单台Merlin 1D发动机动态 (E2)."""

    def __init__(self, engine_id=0):
        self.id = engine_id
        self.thrust_ratio = 0.0       # 实际推力比 [0,1]
        self.thrust = 0.0             # 实际推力 N
        self.thrust_tau = 0.3         # 油门响应时间常数
        self.startup_time = 2.5       # 点火到满推力 2.5s
        self.shutdown_tau = 0.8       # 关机尾推时间常数
        self.startup_progress = 1.0   # 点火进度 [0,1]
        self.thrust_noise_std = 0.03  # 推力抖动 ±3%
        self.is_active = False        # 是否点火

    def update(self, throttle_cmd, h, dt, rng=None):
        """更新发动机推力. throttle_cmd[0,1]直接推力百分比."""
        pct = np.clip(throttle_cmd, 0.0, 1.0)

        if pct > 0.01:
            self.is_active = True
            if self.thrust_ratio < 0.01 and self.startup_progress < 1.0:
                # 点火S曲线
                self.startup_progress = min(1.0, self.startup_progress + dt / self.startup_time)
                t = self.startup_progress
                s_curve = 3 * t * t - 2 * t * t * t
                target = max(pct, 0.4)
                self.thrust_ratio = s_curve * target
            else:
                # 正常运行: 一阶滞后
                alpha = min(dt / self.thrust_tau, 1.0)
                self.thrust_ratio += alpha * (pct - self.thrust_ratio)
                self.startup_progress = 1.0
        else:
            # 关机尾推
            if self.thrust_ratio > 0.01:
                self.thrust_ratio *= np.exp(-dt / self.shutdown_tau)
            else:
                self.thrust_ratio = 0.0
                self.is_active = False
            self.startup_progress = 0.0

        # 推力抖动
        if rng is not None:
            noise = self.thrust_noise_std * rng.standard_normal() * self.thrust_ratio
            ratio_noisy = np.clip(self.thrust_ratio + noise, 0.0, 1.0)
        else:
            ratio_noisy = self.thrust_ratio

        T_max = rp.thrust_at_alt(h)
        self.thrust = ratio_noisy * T_max
        return self.thrust


class Octaweb:
    """9台Merlin 1D发动机矩阵 (Octaweb布局).

    布局: 8台外围(等间距45°) + 1台中心.
    中心发动机(0号): 双向TVC, 提供姿态控制.
    外围发动机(1-8号): 固定推力方向, 仅轴向推力.

    1-3-1策略:
      3发模式: 中心(0) + 外围1,5 (对角, 力矩平衡)
      1发模式: 仅中心(0)
    """

    # 发动机位置 (Yb, Zb平面, 相对几何中心)
    # 中心在原点, 外围在半径R处
    R_ENG = rp.DIAMETER / 2.0 * 0.7  # 发动机布局半径

    ENGINE_POS_YZ = np.array([
        [0.0, 0.0],                          # 0: 中心
        [R_ENG, 0.0],                        # 1: +Y (右)
        [R_ENG * 0.707, R_ENG * 0.707],      # 2: +Y+Z
        [0.0, R_ENG],                        # 3: +Z
        [-R_ENG * 0.707, R_ENG * 0.707],     # 4: -Y+Z
        [-R_ENG, 0.0],                       # 5: -Y (左)
        [-R_ENG * 0.707, -R_ENG * 0.707],    # 6: -Y-Z
        [0.0, -R_ENG],                       # 7: -Z
        [R_ENG * 0.707, -R_ENG * 0.707],     # 8: +Y-Z
    ])

    # 3发模式活跃发动机: 中心 + 1,5 (对角, 推力对称)
    ACTIVE_3 = [0, 1, 5]
    # 1发模式活跃发动机: 仅中心
    ACTIVE_1 = [0]

    def __init__(self):
        self.engines = [EngineDynamics(i) for i in range(9)]
        # 中心发动机TVC (与E2 TVC类一致)
        self.center_tvc = TVC(tau=0.05)
        self.n_active = 1  # 当前活跃发动机数
        self.active_mask = np.zeros(9, dtype=bool)
        self.active_mask[0] = True  # 默认仅中心

    def set_engine_config(self, n_engines):
        """设置活跃发动机数量 (1或3)."""
        if n_engines == 3:
            self.active_mask[:] = False
            for idx in self.ACTIVE_3:
                self.active_mask[idx] = True
            self.n_active = 3
        else:
            self.active_mask[:] = False
            self.active_mask[0] = True
            self.n_active = 1

    def get_thrust_bounds(self, h):
        """返回当前配置的(T_min, T_max)."""
        T_max_single = rp.thrust_at_alt(h)
        T_min_single = rp.THRUST_MIN_PCT * T_max_single
        return self.n_active * T_min_single, self.n_active * T_max_single

    def update(self, throttle_cmd, gimbal_pitch_cmd, gimbal_yaw_cmd,
               phase, h, dt, rng=None):
        """更新所有发动机.
        throttle_cmd: G-FOLD总油门指令 [0,1] (相对于单台T_max)
        gimbal_pitch_cmd, gimbal_yaw_cmd: 中心发动机TVC指令
        返回: (total_thrust, gimbal_pitch, gimbal_yaw)
        """
        # 中心发动机TVC更新
        total_thrust, gp, gy = self.center_tvc.update(
            throttle_cmd, gimbal_pitch_cmd, gimbal_yaw_cmd, phase, h, dt)

        # 中心发动机推力 = TVC输出
        self.engines[0].thrust = total_thrust
        self.engines[0].thrust_ratio = total_thrust / rp.thrust_at_alt(h) if rp.thrust_at_alt(h) > 0 else 0
        self.engines[0].is_active = total_thrust > 100

        # 外围发动机 (仅3发模式活跃)
        outer_thrust = 0.0
        for i in range(1, 9):
            if self.active_mask[i]:
                # 外围发动机油门 = 中心油门 (均分总推力)
                self.engines[i].update(throttle_cmd, h, dt, rng)
                outer_thrust += self.engines[i].thrust
            else:
                # 不活跃发动机: 关机尾推
                if self.engines[i].thrust_ratio > 0.01:
                    self.engines[i].update(0.0, h, dt, rng)
                    outer_thrust += self.engines[i].thrust
                else:
                    self.engines[i].thrust = 0.0

        total = total_thrust + outer_thrust
        return total, gp, gy

    def compute_force_moment(self, gp, gy, cg_x):
        """计算总推力(body系)和总力矩(body系).
        中心发动机: 推力沿TVC偏转方向, 力矩=位置×力
        外围发动机: 推力沿+Xb(固定), 力矩=位置×力(仅Yb/Zb偏心)
        """
        F_total = np.zeros(3)
        M_total = np.zeros(3)
        x_tvc = -rp.LENGTH / 2.0 + 1.0  # 发动机X位置

        # 中心发动机 (TVC偏转)
        if self.engines[0].thrust > 100:
            cp = np.cos(gp)
            F_center = np.array([
                self.engines[0].thrust * cp * np.cos(gy),
                self.engines[0].thrust * cp * np.sin(gy),
                self.engines[0].thrust * (-np.sin(gp))
            ])
            r_center = np.array([x_tvc - cg_x, 0.0, 0.0])
            M_center = np.cross(r_center, F_center)
            F_total += F_center
            M_total += M_center

        # 外围发动机 (固定方向, 沿+Xb)
        for i in range(1, 9):
            if self.engines[i].thrust > 100:
                F_outer = np.array([self.engines[i].thrust, 0.0, 0.0])
                y_i, z_i = self.ENGINE_POS_YZ[i]
                r_outer = np.array([x_tvc - cg_x, y_i, z_i])
                M_outer = np.cross(r_outer, F_outer)
                F_total += F_outer
                M_total += M_outer

        return F_total, M_total


class LandingProfile:
    """1-3-1发动机点火策略.

    E5改进: 基于vz的物理切换 (非tgo), 增大滞回防抖.
      3发: vz>55 (高速制动, T_min=1014kN提供强减速~17m/s²)
      1发: vz<40 (精确着陆, T_min=338kN<重力可hover-slam)
      滞回区: 40<vz<55 (保持当前配置, 防抖)
    最小驻留时间: 2s (防快速切换)

    物理依据:
      3发T_min=1014kN >> 重力363kN → 无法维持恒定vz, 只能制动.
      必须在vz降到40时切换1发, 否则3发继续制动到vz<0(上升).
      1发T_min=338kN < 重力363kN → 可hover-slam精确着陆.
    """

    def __init__(self):
        self.current_config = 3  # 初始3发 (高速制动)
        self.vz_switch_high = 55.0   # 1→3触发: vz>55
        self.vz_switch_low = 40.0    # 3→1触发: vz<40 (与G-FOLD 3发终端vz=40对齐)
        self.min_dwell = 2.0         # 最小驻留时间 s
        self.last_switch_time = -10.0  # 上次切换时间

    def decide_engine_config(self, h, vz, tgo, t=0.0):
        """返回应点火的发动机数量 (1或3)."""
        # 驻留时间检查
        if t - self.last_switch_time < self.min_dwell:
            return self.current_config

        if self.current_config == 3:
            # 3发→1发: vz降到40以下 (与G-FOLD 3发终端vz=40对齐)
            if vz < self.vz_switch_low:
                self.current_config = 1
                self.last_switch_time = t
        else:
            # 1发→3发: vz升到55以上
            if vz > self.vz_switch_high:
                self.current_config = 3
                self.last_switch_time = t

        return self.current_config
