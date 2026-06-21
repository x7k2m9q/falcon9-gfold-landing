"""
火箭物理参数 + 动态质量/质心/转动惯量计算.
坐标系: b系, Xb指向头部. 推力沿+Xb.
质心随燃料消耗沿 -Xb 后移 (方案铁律).
物理依据: 猎鹰9号 LOX 储箱在上部(头部方向), RP-1 在下部. 回收段消耗燃料后,
  头部(LOX)变轻, 质心向尾部 -Xb 后移. 故燃料质心设在 +Xb(前), 干质心在 -Xb(后).
  满载 cg 偏前, 空载 cg 偏后, 消耗过程质心单调后移.
"""
import numpy as np

# 基础参数
DRY_MASS = 22000.0          # 干重 kg
INIT_FUEL = 30000.0         # 初始燃料 kg
G0 = 9.80665

# 推力
THRUST_SL = 845000.0        # 海平面推力 N
THRUST_VAC = 1200000.0      # 真空推力 N
THRUST_MIN_PCT = 0.40       # 最小油门 40% (Step B: 还原Merlin 1D真实节流下限, 与G-FOLD T_MIN_FRAC一致)
TVC_GIMBAL_LIMIT_NON_GFOld = np.radians(3.0)  # 非G-FOLD段TVC限幅3度 (1.1修正)
TVC_GIMBAL_LIMIT_GFOld = np.radians(15.0)     # G-FOLD段允许大偏转
TVC_FIRST_ORDER_TAU = 0.5   # TVC一阶延迟 0.5s

# 几何
LENGTH = 30.0               # 箭体长度 m
DIAMETER = 3.35             # 直径 m
REF_AREA = np.pi * (DIAMETER / 2.0) ** 2  # 参考面积
ENGINE_X = -13.0            # 发动机安装位置Xb坐标(尾部, 几何中心-13m)

# 质心与转动惯量基准 (干质心位置, 沿Xb, 原点在箭体几何中心)
# 设几何中心在长度中点. 干质心偏后(发动机集中尾部).
DRY_CG_X = -2.0             # 干质心相对几何中心的Xb坐标(负=尾部方向)
FUEL_CG_X = 1.5             # 燃料(LOX为主)质心Xb坐标(正=头部方向), 消耗使质心后移
# 干转动惯量 (kg·m^2), 绕b系三轴. Xb为纵轴(小), Yb/Zb为横向(大)
Ixx_dry = 5.0e4
Iyy_dry = 2.5e5
Izz_dry = 2.5e5

# 燃料消耗: 由推力与比冲决定. 简化用等效比冲
ISP_SL = 282.0              # 海平面比冲 s
ISP_VAC = 311.0             # 真空比冲 s


def thrust_at_alt(h):
    """高度插值推力. h<0按海平面."""
    if h <= 0.0:
        return THRUST_SL
    if h >= 70000.0:
        return THRUST_VAC
    return THRUST_SL + (THRUST_VAC - THRUST_SL) * (h / 70000.0)


def isp_at_alt(h):
    if h <= 0.0:
        return ISP_SL
    if h >= 70000.0:
        return ISP_VAC
    return ISP_SL + (ISP_VAC - ISP_SL) * (h / 70000.0)


def mass_properties(fuel_mass):
    """根据当前燃料质量返回 (总质量, 质心Xb坐标, 转动惯量张量I_body[3,3]).
    质心随燃料消耗沿 -Xb 后移(方案要求). 用平行轴定理重算惯量.
    """
    m_total = DRY_MASS + fuel_mass
    # 混合质心
    cg_x = (DRY_MASS * DRY_CG_X + fuel_mass * FUEL_CG_X) / m_total
    # 干惯量从干质心平移到混合质心: I = I_dry + m_dry * d^2 (横向)
    d_dry = DRY_CG_X - cg_x
    Ixx = Ixx_dry  # 纵轴平移不影响
    Iyy = Iyy_dry + DRY_MASS * d_dry * d_dry
    Izz = Izz_dry + DRY_MASS * d_dry * d_dry
    # 燃料视为集中质量(简化), 加到惯量
    d_fuel = FUEL_CG_X - cg_x
    Ixx_f = 0.0  # 燃料纵轴惯量忽略
    Iyy_f = fuel_mass * d_fuel * d_fuel
    Izz_f = fuel_mass * d_fuel * d_fuel
    Ixx += Ixx_f
    Iyy += Iyy_f
    Izz += Izz_f
    I_body = np.diag([Ixx, Iyy, Izz])
    return m_total, cg_x, I_body


def fuel_flow_rate(thrust, h):
    """燃料质量流率 mdot = T / (g0 * Isp)."""
    return thrust / (G0 * isp_at_alt(h))
