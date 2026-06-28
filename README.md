# Falcon 9 G-FOLD Vertical Landing Simulation | 猎鹰9号G-FOLD垂直回收仿真

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![MC Success Rate](https://img.shields.io/badge/MC%20Success-100%25-brightgreen.svg)](#results)
[![Language](https://img.shields.io/badge/README-Chinese%20%7C%20English-red.svg)](#english-version)

---

## 中文版

### 项目简介

本项目完整复刻并工程化实现了SpaceX猎鹰9号（Falcon 9）运载火箭一级垂直回收的制导与控制算法。基于Lars Blackmore等人提出的G-FOLD（Guidance for Fuel-Optimal Divert）二阶锥规划（SOCP）理论框架，构建了包含6自由度刚体动力学、弹性体模态、推进剂晃动、多发动机1-3-1点火剖面、乘性扩展卡尔曼滤波（MEKF）状态估计、Smith预估器延迟补偿及分级异常处理兜底策略的完整仿真系统。

### 核心成果

| 指标 | 结果 |
|------|------|
| 蒙特卡洛成功率 | **100/100 (100%)** |
| 着陆速度 P95 | 3.06 m/s |
| 着陆姿态 P95 | 0.75° |
| 着陆点偏差 P95 | 1.35 m |
| 求解器失败率 | 0.00% (3967次求解) |
| 故障恢复率 | 3/3 (100%) |

### 系统架构

```
FlightController
├── EKF (MEKF, 15维误差状态)
├── DelayCompensator (Smith预估器, 150ms)
├── G-FOLD (SOCP凸优化, CLARABEL求解器)
├── AttitudeController (PD + NotchFilter)
├── Octaweb (9台Merlin 1D, 1-3-1剖面)
├── SafetyMonitor (可达集/燃料/姿态监控)
└── FallbackPD (hover-slam bang-bang兜底)
```

### 关键技术

1. **G-FOLD SOCP制导**：6维状态凸优化，推力幅值SOC约束+指向锥约束+终端软约束
2. **无损凸化验证**：$T_{\min}<g$时gap=1.000（理论成立），$T_{\min}>g$时gap=1.27（需闭环补偿）
3. **多发动机1-3-1剖面**：3发高速制动（$T_{\min}\gg g$）→ 1发精确着陆（$T_{\min}<g$）
4. **Hover-slam控制**：$v_{\text{target}}=2\sqrt{h}$，恒定减速度2 m/s²
5. **MEKF状态估计**：15维误差状态，去上帝视角，末端50m切换纯IMU+雷达
6. **工程硬化**：SafetyMonitor + FlightController分级异常处理 + CLARABEL求解器

### 目录结构

```
falcon9-gfold-landing/
├── src/                        # 源代码
│   ├── dynamics.py             # 6DOF RK4动力学引擎
│   ├── guidance.py             # G-FOLD SOCP制导 + MPC
│   ├── ekf.py                  # 乘性扩展卡尔曼滤波
│   ├── octaweb.py              # 9发动机矩阵 + 1-3-1剖面
│   ├── flight_controller.py    # 统一控制框架 + 异常处理
│   ├── safety_monitor.py       # 安全监控 + 兜底策略
│   ├── attitude_control.py     # 姿态控制 + 陷波滤波器
│   ├── flex_dynamics.py        # 弹性体 + 晃动建模
│   ├── delay_comp.py           # Smith预估器延迟补偿
│   ├── actuators.py            # TVC + GridFin + RCS
│   ├── sensors.py              # IMU + GPS + 雷达高度计
│   ├── rocket_params.py        # 火箭物理参数
│   ├── quaternion_utils.py     # 四元数工具
│   ├── atmosphere.py           # 1976标准大气
│   ├── wind.py                 # Dryden风场
│   └── aero.py                 # 气动力计算
├── tests/                      # 测试脚本
│   ├── e6_hardening.py         # E6工程硬化集成测试
│   ├── mc100_batch.py          # 100次蒙特卡洛测试
│   ├── e5_octaweb.py           # E5多发动机测试
│   ├── e1e3_nominal.py         # E1+E3联合测试
│   ├── e4_delay_comp.py        # E4延迟补偿测试
│   └── step_d_verify.py        # Step D无损凸化验证
├── cpp_sim/                    # C++实时飞控仿真 (嵌入式工程化预研)
│   ├── os/                     # 协作式调度器+SPSC环形缓冲+看门狗
│   ├── hal/                    # 硬件抽象层 (SensorInterface/ActuatorInterface)
│   ├── core/                   # 定点矩阵+四元数工具
│   ├── tests/                  # C++单元测试 + 性能基准
│   └── CMakeLists.txt
├── docs/                       # 文档
│   └── 论文.md                 # 学术论文
├── .gitignore
├── LICENSE
├── README.md                   # 本文件
└── requirements.txt            # Python依赖
```

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行100次蒙特卡洛测试（约16分钟）
python tests/mc100_batch.py

# 运行E6工程硬化测试（含故障注入）
python tests/e6_hardening.py

# 运行无损凸化验证
python tests/step_d_verify.py
```

### 物理参数

| 参数 | 值 |
|------|-----|
| 干质量 | 22000 kg |
| 初始燃料 | 30000 kg |
| 单台推力 | 845 kN (Merlin 1D) |
| 节流下限 | 40% |
| 仿真步长 | 0.01 s |
| G-FOLD周期 | 1 s (MPC) |
| IMU频率 | 100 Hz |
| GPS频率 | 10 Hz |
| 雷达频率 | 50 Hz (h<100m) |

### 理论方案演进

本项目经历三个理论方案阶段：

1. **理论方案1.0**：6DOF物理底座 + 三执行器 + 姿态控制 + G-FOLD制导
2. **理论方案1.1**：油门回归SOCP (Step A) → 真实节流下限40% (Step B) → 气动力进SOCP (Step C) → 无损凸化验证 (Step D)
3. **理论方案2.0**：弹性体+晃动 (E1) → 发动机动态 (E2) → EKF状态估计 (E3) → 延迟补偿 (E4) → 多发动机1-3-1 (E5) → 工程硬化 (E6)

---

### 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{falcon9_gfold_2026,
  title={G-FOLD Convex Optimization Based 6-DOF Simulation and Engineering Hardening Verification for Falcon 9 Vertical Landing Recovery},
  author={x7k2m9q},
  year={2026},
  url={https://github.com/x7k2m9q/falcon9-gfold-landing}
}
```

### 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## English Version

### Project Overview

This project presents a complete replication and engineering hardening of the SpaceX Falcon 9 first-stage vertical landing guidance and control algorithm. Based on the G-FOLD (Guidance for Fuel-Optimal Divert) Second-Order Cone Programming (SOCP) framework proposed by Lars Blackmore et al., a comprehensive simulation system is constructed, incorporating 6-DOF rigid body dynamics, flexible body modes, propellant sloshing, multi-engine 1-3-1 ignition profile, Multiplicative Extended Kalman Filter (MEKF) state estimation, Smith predictor delay compensation, and hierarchical exception handling fallback strategies.

### Key Results

| Metric | Result |
|--------|--------|
| Monte Carlo Success Rate | **100/100 (100%)** |
| Landing Velocity P95 | 3.06 m/s |
| Landing Attitude P95 | 0.75° |
| Landing Point Deviation P95 | 1.35 m |
| Solver Failure Rate | 0.00% (3967 solves) |
| Fault Recovery Rate | 3/3 (100%) |

### System Architecture

```
FlightController
├── EKF (MEKF, 15-D error state)
├── DelayCompensator (Smith predictor, 150ms)
├── G-FOLD (SOCP convex optimization, CLARABEL solver)
├── AttitudeController (PD + NotchFilter)
├── Octaweb (9x Merlin 1D, 1-3-1 profile)
├── SafetyMonitor (reachable set/fuel/attitude monitor)
└── FallbackPD (hover-slam bang-bang fallback)
```

### Key Technologies

1. **G-FOLD SOCP Guidance**: 6-D state convex optimization with thrust magnitude SOC constraints, pointing cone constraints, and terminal soft constraints
2. **Lossless Convexification Verification**: gap=1.000 when $T_{\min}<g$ (theory holds), gap=1.27 when $T_{\min}>g$ (requires closed-loop compensation)
3. **Multi-Engine 1-3-1 Profile**: 3-engine high-speed braking ($T_{\min}\gg g$) → 1-engine precision landing ($T_{\min}<g$)
4. **Hover-slam Control**: $v_{\text{target}}=2\sqrt{h}$, constant deceleration of 2 m/s²
5. **MEKF State Estimation**: 15-D error state, no god-view, switch to pure IMU+radar below 50m
6. **Engineering Hardening**: SafetyMonitor + FlightController hierarchical exception handling + CLARABEL solver

### Directory Structure

```
falcon9-gfold-landing/
├── src/                        # Source code
│   ├── dynamics.py             # 6DOF RK4 dynamics engine
│   ├── guidance.py             # G-FOLD SOCP guidance + MPC
│   ├── ekf.py                  # Multiplicative EKF
│   ├── octaweb.py              # 9-engine matrix + 1-3-1 profile
│   ├── flight_controller.py    # Unified control framework + exception handling
│   ├── safety_monitor.py       # Safety monitoring + fallback strategies
│   ├── attitude_control.py     # Attitude control + notch filter
│   ├── flex_dynamics.py        # Flexible body + sloshing modeling
│   ├── delay_comp.py           # Smith predictor delay compensation
│   ├── actuators.py            # TVC + GridFin + RCS
│   ├── sensors.py              # IMU + GPS + Radar altimeter
│   ├── rocket_params.py        # Rocket physical parameters
│   ├── quaternion_utils.py     # Quaternion utilities
│   ├── atmosphere.py           # 1976 standard atmosphere
│   ├── wind.py                 # Dryden wind field
│   └── aero.py                 # Aerodynamic force calculation
├── tests/                      # Test scripts
│   ├── e6_hardening.py         # E6 engineering hardening integration test
│   ├── mc100_batch.py          # 100-run Monte Carlo test
│   ├── e5_octaweb.py           # E5 multi-engine test
│   ├── e1e3_nominal.py         # E1+E3 joint test
│   ├── e4_delay_comp.py        # E4 delay compensation test
│   └── step_d_verify.py        # Step D lossless convexification verification
├── cpp_sim/                    # C++ real-time flight control simulation (embedded engineering pre-study)
│   ├── os/                     # Cyclic executive + SPSC ring buffer + watchdog
│   ├── hal/                    # Hardware abstraction (SensorInterface/ActuatorInterface)
│   ├── core/                   # Fixed matrix + quaternion utilities
│   ├── tests/                  # C++ unit tests + performance benchmarks
│   └── CMakeLists.txt
├── docs/                       # Documentation
│   └── 论文.md                 # Academic paper (Chinese)
├── .gitignore
├── LICENSE
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run 100-run Monte Carlo test (~16 minutes)
python tests/mc100_batch.py

# Run E6 engineering hardening test (with fault injection)
python tests/e6_hardening.py

# Run lossless convexification verification
python tests/step_d_verify.py
```

### Physical Parameters

| Parameter | Value |
|-----------|-------|
| Dry Mass | 22000 kg |
| Initial Fuel | 30000 kg |
| Single Engine Thrust | 845 kN (Merlin 1D) |
| Throttle Lower Bound | 40% |
| Simulation Step | 0.01 s |
| G-FOLD Period | 1 s (MPC) |
| IMU Frequency | 100 Hz |
| GPS Frequency | 10 Hz |
| Radar Frequency | 50 Hz (h<100m) |

### Theoretical Plan Evolution

This project went through three theoretical plan stages:

1. **Plan 1.0**: 6DOF physics base + three actuators + attitude control + G-FOLD guidance
2. **Plan 1.1**: Throttle回归SOCP (Step A) → Real throttle lower bound 40% (Step B) → Aerodynamic force into SOCP (Step C) → Lossless convexification verification (Step D)
3. **Plan 2.0**: Flexible body+sloshing (E1) → Engine dynamics (E2) → EKF state estimation (E3) → Delay compensation (E4) → Multi-engine 1-3-1 (E5) → Engineering hardening (E6)

---

### Citation

If you use this project in your research, please cite:

```bibtex
@misc{falcon9_gfold_2026,
  title={G-FOLD Convex Optimization Based 6-DOF Simulation and Engineering Hardening Verification for Falcon 9 Vertical Landing Recovery},
  author={x7k2m9q},
  year={2026},
  url={https://github.com/x7k2m9q/falcon9-gfold-landing}
}
```

### License

MIT License - See [LICENSE](LICENSE) for details.

---

## References | 参考文献

1. Blackmore, L. (2016). Autonomous precision landing of space rockets. *Nordita Winter School*, 4(1), 1-17.
2. Açıkmeşe, B., & Ploen, S. R. (2007). Convex programming approach to powered descent guidance for Mars landing. *JGCD*, 30(5), 1353-1366.
3. Szmuk, M., Reynolds, T. P., & Açıkmeşe, B. (2020). Successive convexification for real-time six-degree-of-freedom powered descent guidance. *JGCD*, 43(8), 1439-1455.
4. Lefferts, E. J., Markley, F. L., & Shuster, M. D. (1982). Kalman filtering for spacecraft attitude estimation. *JGCD*, 5(5), 417-429.

---

**Repository**: https://github.com/x7k2m9q/falcon9-gfold-landing  
**Author**: x7k2m9q  
**Completion Date**: June 2026
