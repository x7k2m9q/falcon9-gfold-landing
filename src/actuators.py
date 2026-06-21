"""
三执行器模块: TVC / 气动栅格舵 / RCS.
物理硬解耦: TVC只管轴向推力(+G-FOLD段限幅偏转), 姿态由栅格舵+RCS负责.
所有执行器输出力/力矩(b系), 由 simulator 注入 dynamics.
"""
import numpy as np
from . import rocket_params as rp

# ---------- TVC ----------
class TVC:
    """推力矢量控制 + 发动机动态(E2).
    E2新增: 点火曲线(2.5s S形)+关机尾推(0.8s指数衰减)+推力抖动(±3%)+伺服速率限制(30°/s).
    物理依据: Merlin 1D涡轮泵惯性(油门tau=0.3s), TVC伺服(tau=0.05s+速率限制).
    一阶延迟0.05s(G-FOLD段需快速响应). 非G-FOLD段限幅3度, G-FOLD段15度."""
    def __init__(self, tau=0.05):
        self.tau = tau                  # TVC伺服一阶延迟
        self.gimbal_pitch = 0.0         # 实际偏角(绕Yb), rad
        self.gimbal_yaw = 0.0           # 实际偏角(绕Zb), rad
        self.gimbal_pitch_prev = 0.0    # 上一步偏角(速率限制用)
        self.gimbal_yaw_prev = 0.0
        self.thrust = 0.0               # 实际推力 N
        self.thrust_ratio = 0.0         # 实际推力比 [0,1] (动态模型用)
        self.thrust_tau = 0.3           # 油门响应时间常数(涡轮泵惯性)

        # E2: 发动机动态参数
        self.startup_time = 2.5         # 点火到满推力 2.5s (Merlin 1D)
        self.shutdown_tau = 0.8         # 关机尾推时间常数 0.8s
        self.startup_progress = 1.0     # 点火进度 [0,1], 1=已完成(默认已点火)
        self.thrust_noise_std = 0.03    # 推力抖动 ±3% (涡轮泵脉动)
        self.gimbal_rate_limit = np.radians(30.0)  # TVC伺服速率限制 30°/s

    def update(self, throttle_cmd, gimbal_pitch_cmd, gimbal_yaw_cmd,
               phase, h, dt):
        """throttle_cmd[0,1]直接推力百分比; phase: 'GFOld'或'G-FOLD'; h:高度m. 返回(thrust, gp, gy).
        E2: 推力动态含点火曲线/关机尾推/抖动, TVC含速率限制."""
        pct = np.clip(throttle_cmd, 0.0, 1.0)

        # === E2: 推力动态 ===
        if pct > 0.01:
            # 有油门指令
            if self.thrust_ratio < 0.01 and self.startup_progress < 1.0:
                # 点火曲线: S形上升 3t²-2t³
                self.startup_progress = min(1.0, self.startup_progress + dt / self.startup_time)
                t = self.startup_progress
                s_curve = 3 * t * t - 2 * t * t * t
                target = max(pct, 0.4)  # 点火期至少T_min(40%)
                self.thrust_ratio = s_curve * target
            else:
                # 正常运行: 一阶滞后跟踪油门
                alpha = min(dt / self.thrust_tau, 1.0)
                self.thrust_ratio += alpha * (pct - self.thrust_ratio)
                self.startup_progress = 1.0
        else:
            # 油门指令=0: 关机尾推(指数衰减)
            if self.thrust_ratio > 0.01:
                self.thrust_ratio *= np.exp(-dt / self.shutdown_tau)
            else:
                self.thrust_ratio = 0.0
            self.startup_progress = 0.0

        # 推力抖动 (±3%, 涡轮泵脉动)
        thrust_noise = self.thrust_noise_std * np.random.randn() * self.thrust_ratio
        thrust_ratio_noisy = np.clip(self.thrust_ratio + thrust_noise, 0.0, 1.0)

        # 实际推力(N)
        T_max = rp.thrust_at_alt(h)
        self.thrust = thrust_ratio_noisy * T_max

        # === E2: TVC伺服 (一阶滞后 + 速率限制) ===
        if phase in ('GFOld', 'G-FOLD', 'DEADBAND'):
            lim = rp.TVC_GIMBAL_LIMIT_GFOld
        else:
            lim = rp.TVC_GIMBAL_LIMIT_NON_GFOld
        gp_cmd = np.clip(gimbal_pitch_cmd, -lim, lim)
        gy_cmd = np.clip(gimbal_yaw_cmd, -lim, lim)

        # 一阶滞后
        alpha_g = min(dt / self.tau, 1.0)
        self.gimbal_pitch += (gp_cmd - self.gimbal_pitch) * alpha_g
        self.gimbal_yaw += (gy_cmd - self.gimbal_yaw) * alpha_g

        # 速率限制 30°/s
        max_rate = self.gimbal_rate_limit * dt
        dp = self.gimbal_pitch - self.gimbal_pitch_prev
        dp = np.clip(dp, -max_rate, max_rate)
        self.gimbal_pitch = self.gimbal_pitch_prev + dp
        dy = self.gimbal_yaw - self.gimbal_yaw_prev
        dy = np.clip(dy, -max_rate, max_rate)
        self.gimbal_yaw = self.gimbal_yaw_prev + dy

        self.gimbal_pitch_prev = self.gimbal_pitch
        self.gimbal_yaw_prev = self.gimbal_yaw

        return self.thrust, self.gimbal_pitch, self.gimbal_yaw

    def get_limit(self, phase):
        return rp.TVC_GIMBAL_LIMIT_GFOld if phase == 'GFOld' else rp.TVC_GIMBAL_LIMIT_NON_GFOld


# ---------- 气动栅格舵 ----------
def gridfin_efficiency(M):
    """栅格舵马赫效率系数. 亚音速上升, 跨音速峰, 超音速缓慢衰减."""
    if M < 0.3:
        return 0.2 + 0.8 * (M / 0.3)
    elif M < 1.0:
        return 1.0 + 0.5 * (M - 0.3) / 0.7
    elif M < 1.5:
        return 1.5
    else:
        return 1.5 * np.exp(-(M - 1.5) / 4.0) + 0.5


class GridFin:
    """4片X型栅格舵. 接收归一化力矩指令 cmd=[roll,pitch,yaw]∈[-1,1]^3 (对应b系[X,Y,Z]),
    用直接分配矩阵D映射到4片偏转(±15°), cmd_i=1对应该片满偏.
    力矩 = eff(M) * qdyn * S * Cδ * (B0 @ delta). B0为几何力矩响应矩阵, 行=[X滚转,Y俯仰,Z偏航]."""
    def __init__(self):
        self.S_fin = 1.5
        self.x_fin = -rp.LENGTH / 2.0 + 2.0
        self.R = rp.DIAMETER / 2.0
        self.C_delta = 2.0
        self.rate_limit = np.radians(50.0)
        self.delta_max = np.radians(15.0)
        s = 1.0 / np.sqrt(2.0)
        self._s = s
        # 4片X型位置(径向)与法向
        self.pos_yz = np.array([[+self.R * s, +self.R * s],
                                [-self.R * s, +self.R * s],
                                [-self.R * s, -self.R * s],
                                [+self.R * s, -self.R * s]])
        # 法向取切向(垂直径向, 升力主导方向), 使三轴力矩可控(含滚转)
        self.norm_yz = np.array([[-s, +s],
                                 [-s, -s],
                                 [+s, -s],
                                 [+s, +s]])
        # 直接分配矩阵 D[4,3]: delta = D @ cmd * delta_max
        # 行=片, 列=[roll, pitch, yaw]. 由B0结构推导:
        #   roll:  4片同向  [+1,+1,+1,+1]
        #   pitch: 0,3正 1,2负 [+1,-1,-1,+1]
        #   yaw:   0,1正 2,3负 [+1,+1,-1,-1]
        self.D = np.array([[+1, +1, +1],
                           [+1, -1, +1],
                           [+1, -1, -1],
                           [+1, +1, -1]], dtype=float)
        self.delta = np.zeros(4)

    def _B0(self, cg_x):
        """几何力矩响应矩阵 [3,4]: 单位偏转单位力->力矩. 行=[滚转X,俯仰Y,偏航Z]."""
        rx = self.x_fin - cg_x
        s = self._s
        Rs = self.R * s
        B0 = np.zeros((3, 4))
        for i in range(4):
            py, pz = self.pos_yz[i]
            ny, nz = self.norm_yz[i]
            B0[0, i] = py * nz - pz * ny
            B0[1, i] = -rx * nz
            B0[2, i] = rx * ny
        return B0

    def update(self, cmd, mach, qdyn, cg_x, dt):
        """cmd=[roll,pitch,yaw]∈[-1,1] (对应b系[X,Y,Z]). 返回 M_gridfin_b[3]=[Mx,My,Mz]."""
        B0 = self._B0(cg_x)
        # 直接分配: delta = D @ cmd * delta_max, cmd_i=1对应满偏
        delta_cmd = self.D @ np.asarray(cmd) * self.delta_max
        # 多轴叠加可能超限, 按比例缩放保持方向
        max_abs = np.max(np.abs(delta_cmd))
        if max_abs > self.delta_max:
            delta_cmd *= self.delta_max / max_abs
        # 速率限制
        max_d = self.rate_limit * dt
        self.delta = np.clip(delta_cmd, self.delta - max_d, self.delta + max_d)
        eff = gridfin_efficiency(mach)
        M = (B0 @ self.delta) * (eff * qdyn * self.S_fin * self.C_delta)
        return M

    def max_torque_estimate(self, mach, qdyn, cg_x):
        """单轴满偏(cmd_i=±1)时的最大力矩. 与直接分配一致."""
        eff = gridfin_efficiency(mach)
        B0 = self._B0(cg_x)
        # D@e_i 给出该轴满偏时4片偏转模式, B0@(D@e_i*delta_max) = 该轴力矩
        # 由于D设计使B0@(D@e_i) = 4*|B0_row_i|/4 * sign... 简化: 直接算
        M_max = np.zeros(3)
        for i in range(3):
            cmd_i = np.zeros(3); cmd_i[i] = 1.0
            delta_i = self.D @ cmd_i * self.delta_max
            M_i = B0 @ delta_i
            M_max[i] = np.abs(M_i[i])
        return M_max * eff * qdyn * self.S_fin * self.C_delta


# ---------- RCS ----------
class RCS:
    """12喷口反作用控制. 2000N固定推力(冷气推进器). PWM调制. 阀门延迟0.05s+一阶推力tau=0.03s."""
    def __init__(self):
        self.F = 2000.0                      # 单喷口推力 N (增大, 猎鹰9级冷气推进器)
        self.n_per_axis = 2                  # 每轴2喷口(一对)
        self.x_rcs = -rp.LENGTH / 2.0 + 1.0  # 俯仰/偏航喷口Xb位置
        self.R_roll = rp.DIAMETER / 2.0 + 2.0  # 滚转力臂(增大, 用栅格舵位置)
        self.valve_delay = 0.05              # 阀门延迟 s
        self.thrust_tau = 0.03               # 推力上升 tau
        # 三轴最大力矩
        self.Mmax_pitch = 2 * self.F * abs(self.x_rcs + 2.0)  # cg≈-2时力臂
        self.Mmax_yaw = self.Mmax_pitch
        self.Mmax_roll = 2 * self.F * self.R_roll
        # PWM状态(占空比), 经延迟和一阶
        self._cmd_buf = []  # (time, cmd)
        self.t = 0.0
        self.duty = np.zeros(3)   # 实际占空比(经延迟+一阶)
        self._active_duty = np.zeros(3)   # 当前生效的占空比(延迟后)
        self._active_sign = np.zeros(3)  # 当前生效的方向

    def _max_torque(self, cg_x):
        return np.array([
            2 * self.F * self.R_roll,                      # 滚转 Xb
            2 * self.F * abs(self.x_rcs - cg_x),           # 俯仰 Yb
            2 * self.F * abs(self.x_rcs - cg_x),           # 偏航 Zb
        ])

    def update(self, cmd_torque, cg_x, dt):
        """cmd_torque[3] 期望力矩(b系). 返回实际 M_rcs_b[3]."""
        self.t += dt
        Mmax = self._max_torque(cg_x)
        # PWM占空比 = |cmd|/Mmax, 方向=sign
        duty_cmd = np.clip(np.abs(cmd_torque) / np.maximum(Mmax, 1.0), 0.0, 1.0)
        sign = np.sign(cmd_torque)
        # 阀门延迟: 指令延迟0.05s后生效, 持续直到新指令过期
        self._cmd_buf.append((self.t, duty_cmd, sign))
        delay = self.valve_delay
        while self._cmd_buf and self._cmd_buf[0][0] <= self.t - delay:
            _, self._active_duty, self._active_sign = self._cmd_buf.pop(0)
        duty_target = self._active_duty
        sign_target = self._active_sign
        # 一阶推力上升
        a = min(dt / self.thrust_tau, 1.0)
        self.duty += (duty_target - self.duty) * a
        M = self.duty * Mmax * sign_target
        return M

    def max_torque(self, cg_x):
        return self._max_torque(cg_x)
