"""
姿态控制环 (100Hz 内环).
控制律: 四元数误差 PD (取矢量部分 + sign(qw)处理双覆盖) + 前馈.
  e_q = q_des ⊗ q_actual^{-1}  (双覆盖: w<0取反)
  M_cmd = Kp * e_vec + Kd * e_omega + I @ omega_des_dot
增益物理整定(自适应惯量, 避免手调): Kp = I*wn^2, Kd = 2*zeta*wn*I.
执行器分配(按动压分段, 严格隔离):
  qdyn > 2000 : 高速 栅格舵独占
  500-2000    : 中速 栅格舵优先 + RCS补不足
  < 500       : 低速 RCS独占
  TVC gimbal_cmd 恒为0 (姿态绝不靠TVC, G-FOLD段TVC偏转由制导环单独设置)
防饱和: 力矩超限按比例缩放保持方向.

E1改造: 串联NotchFilterBank到姿态误差通道(pitch/yaw), 防止控制器激励弯曲共振.
  弯曲模态在Yb/Zb方向, 故只过滤e_vec[1]/e_vec[2]和e_omega[1]/e_omega[2].
  滚转通道(Xb)无弯曲耦合, 不过滤.
"""
import numpy as np
from . import quaternion_utils as qu
from . import rocket_params as rp
from .flex_dynamics import NotchFilterBank


class AttitudeController:
    def __init__(self, wn=2 * np.pi * 0.5, zeta=0.9, sample_rate=100.0,
                 use_notch=True):
        self.wn = wn          # 自然频率 rad/s (~0.5Hz)
        self.zeta = zeta
        self.last_M_cmd = np.zeros(3)
        self.last_gf_cmd = np.zeros(3)
        self.last_rcs_cmd = np.zeros(3)
        # E1: 陷波器组(每通道独立IIR状态). pitch(Yb)和yaw(Zb)各一组.
        self.use_notch = use_notch
        if use_notch:
            self.notch_pitch = NotchFilterBank(sample_rate=sample_rate)
            self.notch_yaw = NotchFilterBank(sample_rate=sample_rate)
            # 角速度误差也加陷波(同一通道, 独立状态)
            self.notch_pitch_omega = NotchFilterBank(sample_rate=sample_rate)
            self.notch_yaw_omega = NotchFilterBank(sample_rate=sample_rate)

    def compute_torque(self, q_des, omega_des, q_actual, omega_actual, I_body,
                       omega_des_dot=None, M_disturbance=None, phase=None):
        """返回期望力矩 M_cmd(b系).
        四元数PD: e_vec ≈ θ_err/2 (矢量部分), 故 Kp=2*I*wn² 补偿1/2因子,
        使闭环方程为标准 θ¨+2ζω_n·θ˙+ω_n²·θ=ω_n²·θ_des.
        M_disturbance: 已知外力矩(如气动力矩), 前馈补偿: M_cmd = PD - M_dist.
        phase: 'G-FOLD'/'DEADBAND'时忽略滚转误差(修复7).
        修复10: 误差在body系计算 e_q=q_actual^{-1}⊗q_des, 而非q_des系.
          旧版 e_q=q_des⊗q_actual^{-1} 给出q_des系误差, 当q_des=Q_VERT(90°旋转)
          时e_vec分量对应[偏航,俯仰,滚转]而非[滚转,俯仰,偏航], 导致:
          ① 零e_vec[0]实际零偏航(非滚转), 偏航倾斜无法修正
          ② M_cmd[0]拿偏航误差送RCS(滚转), M_cmd[2]拿滚转误差送TVC(偏航)"""
        if omega_des_dot is None:
            omega_des_dot = np.zeros(3)
        if M_disturbance is None:
            M_disturbance = np.zeros(3)
        # 修复10: body系误差 e_q = q_actual^{-1} ⊗ q_des
        # e_vec[0]=滚转, e_vec[1]=俯仰, e_vec[2]=偏航 (均为body系)
        e_q = qu.quat_multiply(qu.quat_inverse(q_actual), q_des)
        if e_q[0] < 0.0:
            e_q = -e_q
        e_vec = e_q[1:4].copy()
        # G-FOLD/DEADBAND段: 忽略滚转误差(修复7). 原因: q_des_from_thrust_dir的
        # 隐含滚转角会随推力方向变化而跳变, 导致RCS追逐滚转发散. TVC可独立控制
        # pitch/yaw不受滚转影响, 滚转在DEADBAND后期或着陆后再修正.
        if phase in ('G-FOLD', 'DEADBAND'):
            e_vec[0] = 0.0  # body系[0]=滚转
        e_omega = omega_des - omega_actual
        # E1: 陷波器过滤pitch/yaw误差通道, 防止激励弯曲共振(ω1=2.35Hz, ω2=14.7Hz).
        # 滚转通道(Xb)无弯曲耦合, 不过滤. 注意: 陷波器有相位延迟, 仅在误差通道使用,
        # 不影响稳态精度(陷波器在DC处增益=1).
        if self.use_notch:
            e_vec[1] = self.notch_pitch.filter(e_vec[1])
            e_vec[2] = self.notch_yaw.filter(e_vec[2])
            e_omega[1] = self.notch_pitch_omega.filter(e_omega[1])
            e_omega[2] = self.notch_yaw_omega.filter(e_omega[2])
        # 增益(对角向量, 自适应惯量). Kp=2*I*wn² 补偿 e_vec=θ/2
        i_diag = np.diag(I_body)
        Kp = 2.0 * i_diag * (self.wn ** 2)       # [3]
        Kd = 2.0 * self.zeta * self.wn * i_diag  # [3]
        M_cmd = Kp * e_vec + Kd * e_omega + I_body @ omega_des_dot - M_disturbance
        return M_cmd

    def allocate(self, M_cmd, mach, qdyn, cg_x, gf, rcs, phase=None):
        """按动压分段分配到栅格舵(归一化cmd)和RCS(力矩cmd). 返回 (gf_cmd, rcs_cmd).
        phase='G-FOLD'时: pitch/yaw力矩由TVC承担, 栅格舵/RCS只管滚转."""
        gf_max = gf.max_torque_estimate(mach, qdyn, cg_x)   # [3]
        rcs_max = rcs.max_torque(cg_x)                       # [3]
        gf_max = np.maximum(gf_max, 1.0)
        rcs_max = np.maximum(rcs_max, 1.0)
        gf_cmd = np.zeros(3)
        rcs_cmd = np.zeros(3)

        if phase in ('G-FOLD', 'DEADBAND'):
            # G-FOLD/DEADBAND段(修复3+6): GridFin强制置零, Pitch/Yaw只给TVC, Roll只给RCS
            # DEADBAND段也需TVC姿态控制, 否则低空翻车
            gf_cmd = np.zeros(3)
            rcs_cmd[0] = np.clip(M_cmd[0] / rcs_max[0], -1.0, 1.0) * rcs_max[0]
        elif qdyn > 2000.0:
            gf_cmd = np.clip(M_cmd / gf_max, -1.0, 1.0)
        elif qdyn > 500.0:
            gf_cmd = np.clip(M_cmd / gf_max, -1.0, 1.0)
            M_gf = gf_cmd * gf_max
            rcs_cmd = M_cmd - M_gf
            rcs_cmd = np.clip(rcs_cmd / rcs_max, -1.0, 1.0) * rcs_max
        else:
            rcs_cmd = np.clip(M_cmd / rcs_max, -1.0, 1.0) * rcs_max
        return gf_cmd, rcs_cmd

    def update(self, q_des, omega_des, state, mach, qdyn, cg_x, I_body,
               gf, rcs, dt, omega_des_dot=None, M_disturbance=None,
               phase=None, tvc=None, thrust_actual=0.0):
        """完整一步: 算力矩->分配. 返回 (gf_cmd, rcs_cmd, tvc_gimbal_cmd).
        phase='G-FOLD'时TVC参与姿态控制(低动压下栅格舵/RCS力矩不足).
        tvc: TVC对象(用于获取gimbal限幅). thrust_actual: 当前推力(N).
        M_disturbance: 已知外力矩(b系, 如气动力矩), 前馈补偿."""
        q_actual = state[6:10]
        omega_actual = state[10:13]
        M_cmd = self.compute_torque(q_des, omega_des, q_actual, omega_actual,
                                    I_body, omega_des_dot, M_disturbance, phase=phase)
        gf_cmd, rcs_cmd = self.allocate(M_cmd, mach, qdyn, cg_x, gf, rcs, phase=phase)
        self.last_M_cmd = M_cmd
        self.last_gf_cmd = gf_cmd
        self.last_rcs_cmd = rcs_cmd

        # TVC gimbal: G-FOLD/DEADBAND段用TVC做姿态控制(推力偏转产生巨大力矩)
        tvc_gimbal_cmd = np.zeros(2)  # [pitch, yaw]
        if phase in ('G-FOLD', 'DEADBAND') and tvc is not None and thrust_actual > 1000.0:
            # TVC力矩 = 推力 * sin(gimbal) * arm
            # arm = 发动机到CG距离
            arm = abs(cg_x - rp.ENGINE_X)
            # dynamics: M = r_tvc × F_thrust, r_tvc=[x_tvc-cg_x,0,0](负,指向尾部)
            # np.cross([rx,0,0],[Fx,Fy,Fz]) = [0, -rx*Fz, rx*Fy]
            # 正gp => Fz=-T*sin(gp)<0 => M_y=-rx*Fz=-负*负=负. 故正gp产生负M_y.
            # 要产生正M_cmd[1], 需gp<0. gp = -arcsin(M_y/(T*arm))... 不对!
            # 实测: 正gp=0.1 -> M_y=-1433. 故 gp = arcsin(-M_y/(T*arm)) = -arcsin(M_y/(T*arm))
            # 即 gp 和 M_y 反号: gp = -arcsin(M_y/denom) => M_y>0时gp<0, 产生M_y=-rx*Fz>0? 
            # 验证: gp<0 => Fz=-T*sin(gp)>0 => M_y=-rx*Fz=-负*正=正. 对!
            M_y = M_cmd[1]  # pitch
            M_z = M_cmd[2]  # yaw
            denom = thrust_actual * arm
            if denom > 1.0:
                # gp与M_y反号: 正M_y需负gp
                gp = -np.arcsin(np.clip(M_y / denom, -1.0, 1.0))
                # gy: 正gy => Fy=T*cos*sin(gy)>0 => M_z=rx*Fy=负*正=负. 故正gy产生负M_z.
                # 要产生正M_z, 需gy<0. gy = -arcsin(M_z/denom)
                gy = -np.arcsin(np.clip(M_z / denom, -1.0, 1.0))
                lim = rp.TVC_GIMBAL_LIMIT_GFOld
                tvc_gimbal_cmd = np.array([np.clip(gp, -lim, lim),
                                           np.clip(gy, -lim, lim)])

        return gf_cmd, rcs_cmd, tvc_gimbal_cmd

    def reset_notch(self):
        """重置陷波器内部状态(用于仿真重置)."""
        if self.use_notch:
            self.notch_pitch.reset()
            self.notch_yaw.reset()
            self.notch_pitch_omega.reset()
            self.notch_yaw_omega.reset()


def q_des_from_tilt(tilt_rad, axis_body=np.array([0, 1, 0])):
    """构造期望姿态: 垂直姿态绕body轴倾斜tilt_rad. q_des = Q_VERT ⊗ q_tilt."""
    axis = axis_body / np.linalg.norm(axis_body)
    q_tilt = np.array([np.cos(tilt_rad / 2),
                       axis[0] * np.sin(tilt_rad / 2),
                       axis[1] * np.sin(tilt_rad / 2),
                       axis[2] * np.sin(tilt_rad / 2)])
    return qu.quat_multiply(qu.Q_VERT, q_tilt)
