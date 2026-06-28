// =============================================================================
// ekf.hpp - 15维误差状态乘性扩展卡尔曼滤波器 (MEKF)
// 猎鹰9号火箭回收算法 C++ 翻译项目
// 对应 Python src/ekf.py
//
// 设计依据: 理论方案2.0 E3节, "去掉上帝视角"
//   - 标准航天做法: 误差状态用3维罗德里格斯参数, 避免四元数过参数化导致的协方差奇异
//   - 全状态(16维): p_n[3] + v_n[3] + q_bn[4] + b_g[3] + b_a[3]
//   - 误差状态(15维): δp[3] + δv[3] + δθ[3] + δb_g[3] + δb_a[3]
//
// 坐标系:
//   n: NED (X北, Y东, Z地下, 重力+Z)
//   b: body (Xb头部, Yb右, Zb=Xb×Yb)
//   q=[w,x,y,z] 表示 b->n 旋转 (Hamilton 约定)
//
// 约法三章:
//   1. 零动态内存 (无 new/malloc/vector, 全部定长数组)
//   2. float32 默认, double 仅用于 EKF 协方差 P 及相关矩阵运算
//   3. 算法保真: 严格对应 Python src/ekf.py
// =============================================================================
#pragma once

#include <cmath>
#include <cstdint>

#include "../core/types.hpp"

namespace falcon9 {

// ---------------------------------------------------------------------------
// 通用矩阵运算辅助函数 (行主序, 零动态内存)
// 命名: mat_<op>_<dims>, 通用版本用 mat_<op>
// ---------------------------------------------------------------------------

// C[m×n] = A[m×n] + B[m×n]
inline void mat_add(const double* A, const double* B, double* C, int m, int n) {
    for (int i = 0; i < m * n; ++i) C[i] = A[i] + B[i];
}

// C[m×n] = A[m×k] * B[k×n]  (行主序)
inline void mat_mul(const double* A, const double* B, double* C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// AT[n×m] = A[m×n]^T  (行主序)
inline void mat_transpose(const double* A, double* AT, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            AT[j * m + i] = A[i * n + j];
        }
    }
}

// M = (M + M^T) / 2  (方阵 n×n 就地对称化)
inline void mat_symmetrize(double* M, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double avg = 0.5 * (M[i * n + j] + M[j * n + i]);
            M[i * n + j] = avg;
            M[j * n + i] = avg;
        }
    }
}

// 6x6 矩阵求逆 (Gauss-Jordan 消元 + 部分主元选取)
// 返回 true 表示成功, false 表示奇异 (S_inv 置零)
// 用于 GPS 更新中 S = H P H^T + R 的求逆
inline bool invert_6x6(const double S[36], double S_inv[36]) {
    // 增广矩阵 [S | I], 行主序 [6][12]
    double aug[6][12];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) aug[i][j] = S[i * 6 + j];
        for (int j = 0; j < 6; ++j) aug[i][j + 6] = (i == j) ? 1.0 : 0.0;
    }

    for (int col = 0; col < 6; ++col) {
        // 选取主元 (部分主元法)
        int pivot = col;
        double max_val = std::fabs(aug[col][col]);
        for (int row = col + 1; row < 6; ++row) {
            double v = std::fabs(aug[row][col]);
            if (v > max_val) { max_val = v; pivot = row; }
        }
        if (max_val < 1e-15) {
            // 奇异矩阵, 返回零矩阵 (嵌入式环境不抛异常)
            for (int i = 0; i < 36; ++i) S_inv[i] = 0.0;
            return false;
        }

        // 交换行
        if (pivot != col) {
            for (int j = 0; j < 12; ++j) {
                double tmp = aug[col][j];
                aug[col][j] = aug[pivot][j];
                aug[pivot][j] = tmp;
            }
        }

        // 消去其他行
        double inv_diag = 1.0 / aug[col][col];
        for (int row = 0; row < 6; ++row) {
            if (row == col) continue;
            double factor = aug[row][col] * inv_diag;
            if (factor == 0.0) continue;
            for (int j = col; j < 12; ++j) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // 归一化对角线, 提取逆矩阵
    for (int i = 0; i < 6; ++i) {
        double diag = aug[i][i];
        double inv = 1.0 / diag;
        for (int j = 0; j < 6; ++j) {
            S_inv[i * 6 + j] = aug[i][j + 6] * inv;
        }
    }
    return true;
}

// ===========================================================================
// MEKF - 15维误差状态乘性扩展卡尔曼滤波器
//
// 状态顺序: [δp(3), δv(3), δθ(3), δbg(3), δba(3)]
// 标称状态: p[3], v[3], q[4], bg[3], ba[3] (float32)
// 协方差:   P[15×15] (double, 行主序, 数值稳定)
// ===========================================================================
class MEKF {
public:
    // === 标称状态 (float32) ===
    float p[3];       // NED 位置 [m]
    float v[3];       // NED 速度 [m/s]
    float q[4];       // 四元数 [w,x,y,z] Hamilton, b->n
    float bg[3];      // 陀螺零偏 [rad/s]
    float ba[3];      // 加速度计零偏 [m/s²]

    // === 误差状态协方差 (double, 15×15, 行主序) ===
    // 状态顺序: [δp, δv, δθ, δbg, δba]
    double P[225];

    // === 诊断统计 (float32, 用于监控) ===
    float last_innovation_gps[6];    // GPS 新息 (6维)
    float last_innovation_radar;     // 雷达新息 (1维)
    float last_K_gps_norm;           // GPS 卡尔曼增益范数
    float last_K_radar_norm;         // 雷达卡尔曼增益范数

    // -----------------------------------------------------------------------
    // 默认构造: 零状态 + 单位四元数 + 默认 P/Q
    // -----------------------------------------------------------------------
    MEKF() {
        float zero[3] = {0.0f, 0.0f, 0.0f};
        float q_id[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        reset(zero, zero, q_id, zero, zero, 0.01f);
        // 构造时使用 __init__ 的 P 初值 (与 reset 略有不同, 保真 Python)
        init_P_construct();
    }

    // -----------------------------------------------------------------------
    // 重置滤波器状态 (简化版本, 零偏=0, dt=0.01)
    // -----------------------------------------------------------------------
    void reset(const float pos0[3], const float vel0[3], const float q0[4]) {
        float zero[3] = {0.0f, 0.0f, 0.0f};
        reset(pos0, vel0, q0, zero, zero, 0.01f);
    }

    // -----------------------------------------------------------------------
    // 重置滤波器状态
    // 对应 Python ekf.py reset()
    // -----------------------------------------------------------------------
    void reset(const float pos0[3], const float vel0[3], const float q0[4],
               const float bg0[3], const float ba0[3], float dt) {
        // 标称状态
        for (int i = 0; i < 3; ++i) {
            p[i]  = pos0[i];
            v[i]  = vel0[i];
            bg[i] = bg0[i];
            ba[i] = ba0[i];
        }
        for (int i = 0; i < 4; ++i) q[i] = q0[i];
        quat_normalize(q);

        // 协方差 P 初值 (reset 版本)
        init_P_reset();

        // 过程噪声 Q (离散化, 与 Python __init__ 一致)
        init_Q(dt);

        // 测量噪声参数
        R_gps_pos = 0.5f * 0.5f;       // 0.25 m²
        R_gps_vel = 0.1f * 0.1f;       // 0.01 (m/s)²
        R_radar   = 0.05f * 0.05f;     // 0.0025 m²

        // 重力 (NED, +Z 向下)
        g_n[0] = 0.0f;
        g_n[1] = 0.0f;
        g_n[2] = G0;

        // 自适应 GPS 噪声因子
        gps_noise_scale = 1.0f;

        // 末端模式 (h<50m 切纯 IMU+雷达)
        terminal_mode = false;

        // 诊断统计清零
        for (int i = 0; i < 6; ++i) last_innovation_gps[i] = 0.0f;
        last_innovation_radar = 0.0f;
        last_K_gps_norm       = 0.0f;
        last_K_radar_norm     = 0.0f;
    }

    // -----------------------------------------------------------------------
    // 预测步 (IMU 驱动, 100Hz)
    // 对应 Python ekf.py predict()
    //   gyro_meas:  陀螺测量 (b系角速度) [rad/s]
    //   accel_meas: 加计测量 (b系比力, 已去重力) [m/s²]
    //   dt:         时间步长 [s]
    // -----------------------------------------------------------------------
    void predict(const float gyro_meas[3], const float accel_meas[3], float dt) {
        // 校正零偏
        float omega_b[3], f_b[3];
        for (int i = 0; i < 3; ++i) {
            omega_b[i] = gyro_meas[i] - bg[i];
            f_b[i]     = accel_meas[i] - ba[i];  // 比力 (specific force)
        }

        // 旋转矩阵 C_bn
        float C_bn[9];
        quat_to_rotmat(q, C_bn);

        // === 标称状态传播 ===
        // 位置: p_dot = v  →  p += v * dt
        for (int i = 0; i < 3; ++i) p[i] += v[i] * dt;

        // 速度: v_dot = C_bn @ f_b + g_n  →  v += a_n * dt
        float a_n[3];
        mat3_vec_mul(C_bn, f_b, a_n);
        for (int i = 0; i < 3; ++i) a_n[i] += g_n[i];
        for (int i = 0; i < 3; ++i) v[i] += a_n[i] * dt;

        // 姿态: q_dot = 0.5 * q ⊗ [0, omega_b]  →  q += q_dot * dt, 归一化
        float q_dot[4];
        quat_kinematics(q, omega_b, q_dot);
        for (int i = 0; i < 4; ++i) q[i] += q_dot[i] * dt;
        quat_normalize(q);
        // 零偏保持不变 (随机游走由 Q 处理)

        // === 误差状态协方差传播 ===
        // F 矩阵 (15×15, 连续时间)
        //   δp_dot = δv
        //   δv_dot = -C_bn @ [f_b]× @ δθ - C_bn @ δba
        //   δθ_dot = -[omega_b]× @ δθ - δbg
        double F[225];
        for (int i = 0; i < 225; ++i) F[i] = 0.0;

        // F[0:3, 3:6] = I3  (δp_dot = δv)
        F[0 * 15 + 3] = 1.0;
        F[1 * 15 + 4] = 1.0;
        F[2 * 15 + 5] = 1.0;

        // F[3:6, 6:9] = -C_bn @ skew(f_b)  (δv_dot / δθ)
        float skew_fb[9], tmp3x3[9];
        skew(f_b, skew_fb);
        mat3_mul(C_bn, skew_fb, tmp3x3);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                F[(3 + i) * 15 + (6 + j)] = -static_cast<double>(tmp3x3[i * 3 + j]);

        // F[3:6, 12:15] = -C_bn  (δv_dot / δba)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                F[(3 + i) * 15 + (12 + j)] = -static_cast<double>(C_bn[i * 3 + j]);

        // F[6:9, 6:9] = -skew(omega_b)  (δθ_dot / δθ)
        float skew_omega[9];
        skew(omega_b, skew_omega);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                F[(6 + i) * 15 + (6 + j)] = -static_cast<double>(skew_omega[i * 3 + j]);

        // F[6:9, 9:12] = -I3  (δθ_dot / δbg)
        F[6 * 15 + 9]  = -1.0;
        F[7 * 15 + 10] = -1.0;
        F[8 * 15 + 11] = -1.0;

        // 离散化: Phi = I + F*dt (一阶, 足够小步长)
        double Phi[225];
        for (int i = 0; i < 225; ++i) Phi[i] = F[i] * static_cast<double>(dt);
        for (int i = 0; i < 15; ++i) Phi[i * 15 + i] += 1.0;

        // 协方差传播: P = Phi @ P @ Phi^T + Q  (Q 已是离散化, 不再乘 dt)
        double PhiT[225], tmp1[225], tmp2[225];
        mat_transpose(Phi, PhiT, 15, 15);
        mat_mul(Phi, P, tmp1, 15, 15, 15);
        mat_mul(tmp1, PhiT, tmp2, 15, 15, 15);
        mat_add(tmp2, Q, P, 15, 15);

        // 对称化 + 保证正定
        mat_symmetrize(P, 15);

        // 自适应 GPS 噪声: 检测高动态
        float a_mag = std::sqrt(a_n[0] * a_n[0] + a_n[1] * a_n[1] + a_n[2] * a_n[2]);
        if (a_mag > 5.0f * G0) {  // >5g
            gps_noise_scale = 10.0f;
        } else {
            gps_noise_scale = 1.0f;  // 平滑回归
        }
    }

    // -----------------------------------------------------------------------
    // GPS 更新 (10Hz, 位置+速度, 6维)
    // 对应 Python ekf.py update_gps()
    //   pos_meas, vel_meas: NED 系测量值 [3] each
    // 末端模式 (h<50m) 下跳过 GPS 更新
    // -----------------------------------------------------------------------
    void update_gps(const float pos_meas[3], const float vel_meas[3]) {
        if (terminal_mode) return;  // 末端不信任 GPS

        // 新息 y = z - h_pred = [pos_meas - p; vel_meas - v]
        double y[6];
        for (int i = 0; i < 3; ++i) {
            y[i]     = static_cast<double>(pos_meas[i]) - p[i];
            y[3 + i] = static_cast<double>(vel_meas[i]) - v[i];
        }
        for (int i = 0; i < 6; ++i) last_innovation_gps[i] = static_cast<float>(y[i]);

        // H 矩阵 (6×15): [I3 0 0 0 0; 0 I3 0 0 0]
        double H[6 * 15];
        for (int i = 0; i < 90; ++i) H[i] = 0.0;
        H[0 * 15 + 0] = 1.0; H[1 * 15 + 1] = 1.0; H[2 * 15 + 2] = 1.0;
        H[3 * 15 + 3] = 1.0; H[4 * 15 + 4] = 1.0; H[5 * 15 + 5] = 1.0;

        // 自适应测量噪声 R (6×6, 对角)
        double R[36];
        for (int i = 0; i < 36; ++i) R[i] = 0.0;
        double r_pos = static_cast<double>(R_gps_pos) * gps_noise_scale;
        double r_vel = static_cast<double>(R_gps_vel) * gps_noise_scale;
        R[0]  = r_pos; R[7]  = r_pos; R[14] = r_pos;
        R[21] = r_vel; R[28] = r_vel; R[35] = r_vel;

        // S = H @ P @ H^T + R  (6×6)
        double HT[15 * 6], tmp1[6 * 15], tmp2[6 * 6], S[36];
        mat_transpose(H, HT, 6, 15);
        mat_mul(H, P, tmp1, 6, 15, 15);    // 6×15
        mat_mul(tmp1, HT, tmp2, 6, 15, 6); // 6×6
        mat_add(tmp2, R, S, 6, 6);

        // K = P @ H^T @ inv(S)  (15×6)
        double S_inv[36];
        if (!invert_6x6(S, S_inv)) {
            return;  // 奇异, 跳过本次更新
        }
        double tmp3[15 * 6];
        mat_mul(P, HT, tmp3, 15, 15, 6);   // 15×6
        double K[15 * 6];
        mat_mul(tmp3, S_inv, K, 15, 6, 6); // 15×6

        // K 范数 (诊断)
        double k_norm = 0.0;
        for (int i = 0; i < 90; ++i) k_norm += K[i] * K[i];
        last_K_gps_norm = static_cast<float>(std::sqrt(k_norm));

        // 状态修正: dx = K @ y  (15维误差状态)
        double dx[15];
        mat_mul(K, y, dx, 15, 6, 1);

        // 误差状态注入
        inject_error(dx);

        // 协方差更新 (Joseph form, 保证正定)
        // I_KH = I - K @ H  (15×15)
        double KH[15 * 15], I_KH[225];
        mat_mul(K, H, KH, 15, 6, 15);
        for (int i = 0; i < 225; ++i) I_KH[i] = -KH[i];
        for (int i = 0; i < 15; ++i) I_KH[i * 15 + i] += 1.0;

        // P = I_KH @ P @ I_KH^T + K @ R @ K^T
        double I_KH_T[225], tmp4[225], tmp5[225];
        double KR[15 * 6], KRKt[225], Kt[6 * 15];
        mat_transpose(I_KH, I_KH_T, 15, 15);
        mat_mul(I_KH, P, tmp4, 15, 15, 15);
        mat_mul(tmp4, I_KH_T, tmp5, 15, 15, 15);

        mat_mul(K, R, KR, 15, 6, 6);     // 15×6
        mat_transpose(K, Kt, 15, 6);     // 6×15
        mat_mul(KR, Kt, KRKt, 15, 6, 15); // 15×15

        mat_add(tmp5, KRKt, P, 15, 15);
        mat_symmetrize(P, 15);
    }

    // -----------------------------------------------------------------------
    // 雷达高度计更新 (50Hz, h<100m, 1维)
    // 对应 Python ekf.py update_radar()
    //   alt_meas: 高度 (向上为正, m)
    // 关键: H[0,2] = -1.0 (NED Z-down, 高度 = -z)
    // -----------------------------------------------------------------------
    void update_radar(float alt_meas) {
        // 测量: z = alt, 预测测量: h_pred = -p[2] (高度 = -Z)
        double y = static_cast<double>(alt_meas) - (-static_cast<double>(p[2]));
        last_innovation_radar = static_cast<float>(y);

        // H 矩阵 (1×15): dh/dp[2] = -1.0
        // 修复 (Python 注释): 原为 +1.0 导致符号错误, 雷达说"更低"时 EKF 反而认为"更高"
        double H[15];
        for (int i = 0; i < 15; ++i) H[i] = 0.0;
        H[2] = -1.0;  // δz = -δp[2] (NED Z=地下, 高度=-Z)

        double R = static_cast<double>(R_radar);

        // S = H @ P @ H^T + R  (1×1 标量)
        // H @ P: 1×15
        double HP[15];
        mat_mul(H, P, HP, 1, 15, 15);
        // H @ P @ H^T: 1×1
        double S = 0.0;
        for (int i = 0; i < 15; ++i) S += HP[i] * H[i];
        S += R;

        if (std::fabs(S) < 1e-15) return;  // 奇异保护

        // K = P @ H^T / S  (15×1)
        // H^T[2,0] = H[0,2] = -1.0, 故 K[i] = P[i,2] * (-1.0) / S
        double K[15];
        for (int i = 0; i < 15; ++i) {
            K[i] = P[i * 15 + 2] * H[2] / S;
        }

        // K 范数 (诊断)
        double k_norm = 0.0;
        for (int i = 0; i < 15; ++i) k_norm += K[i] * K[i];
        last_K_radar_norm = static_cast<float>(std::sqrt(k_norm));

        // dx = K * y  (15维)
        double dx[15];
        for (int i = 0; i < 15; ++i) dx[i] = K[i] * y;

        // 误差状态注入
        inject_error(dx);

        // 协方差更新 (Joseph form)
        // I_KH = I - K @ H  (15×15),  K@H: 15×1 * 1×15 = 15×15
        double I_KH[225];
        for (int i = 0; i < 15; ++i)
            for (int j = 0; j < 15; ++j)
                I_KH[i * 15 + j] = -K[i] * H[j];
        for (int i = 0; i < 15; ++i) I_KH[i * 15 + i] += 1.0;

        // P = I_KH @ P @ I_KH^T + K * R * K^T
        double I_KH_T[225], tmp1[225], tmp2[225], KRKt[225];
        mat_transpose(I_KH, I_KH_T, 15, 15);
        mat_mul(I_KH, P, tmp1, 15, 15, 15);
        mat_mul(tmp1, I_KH_T, tmp2, 15, 15, 15);

        for (int i = 0; i < 15; ++i)
            for (int j = 0; j < 15; ++j)
                KRKt[i * 15 + j] = K[i] * R * K[j];

        mat_add(tmp2, KRKt, P, 15, 15);
        mat_symmetrize(P, 15);
    }

    // -----------------------------------------------------------------------
    // 模式切换: 末端模式 (h<50m) 禁用 GPS, 仅用 IMU+雷达
    // -----------------------------------------------------------------------
    void set_terminal_mode(bool enabled) { terminal_mode = enabled; }
    bool is_terminal_mode() const { return terminal_mode; }

    // -----------------------------------------------------------------------
    // 获取状态估计到 StateEstimate 结构体
    // -----------------------------------------------------------------------
    void get_state(StateEstimate& state) const {
        for (int i = 0; i < 3; ++i) {
            state.p[i] = p[i];
            state.v[i] = v[i];
            state.bg[i] = bg[i];
            state.ba[i] = ba[i];
            state.omega[i] = 0.0f;
        }
        for (int i = 0; i < 4; ++i) state.q[i] = q[i];
        for (int i = 0; i < 225; ++i) state.P[i] = P[i];
    }

    // -----------------------------------------------------------------------
    // 输出: 标称状态 + 不确定度 (3σ)
    //   state[13] = [p_n(3), v_n(3), q(4), omega_b(3)]
    //   sigma[15] = sqrt(diag(P))
    // 注意: omega_b 占位为 0, 由调用方用 IMU 测量 - bg 填充
    // -----------------------------------------------------------------------
    void get_state(float state[13], float sigma[15]) const {
        for (int i = 0; i < 3; ++i) state[i] = p[i];
        for (int i = 0; i < 3; ++i) state[3 + i] = v[i];
        for (int i = 0; i < 4; ++i) state[6 + i] = q[i];
        // omega_b 占位, 由调用方填充
        state[10] = 0.0f;
        state[11] = 0.0f;
        state[12] = 0.0f;

        if (sigma) {
            for (int i = 0; i < 15; ++i) {
                double v = P[i * 15 + i];
                sigma[i] = (v > 0.0) ? static_cast<float>(std::sqrt(v)) : 0.0f;
            }
        }
    }

    // 返回校正后的角速度估计 = gyro_meas - bg
    void get_estimated_omega(const float gyro_meas[3], float omega[3]) const {
        for (int i = 0; i < 3; ++i) omega[i] = gyro_meas[i] - bg[i];
    }

    // 返回校正后的比力估计 = accel_meas - ba
    void get_estimated_accel(const float accel_meas[3], float accel[3]) const {
        for (int i = 0; i < 3; ++i) accel[i] = accel_meas[i] - ba[i];
    }

private:
    // === 过程噪声 (15×15, double, 离散化) ===
    double Q[225];

    // === 测量噪声参数 ===
    float R_gps_pos;    // GPS 位置噪声方差 [m²]
    float R_gps_vel;    // GPS 速度噪声方差 [(m/s)²]
    float R_radar;      // 雷达高度噪声方差 [m²]

    // === 重力 (NED, +Z 向下) ===
    float g_n[3];

    // === 自适应 GPS 噪声因子 (高动态时放大) ===
    float gps_noise_scale;

    // === 末端模式标志 (h<50m 切纯 IMU+雷达) ===
    bool terminal_mode;

    // -----------------------------------------------------------------------
    // 初始化过程噪声 Q (离散化)
    // 对应 Python ekf.py __init__ 中 Q 的构造
    // -----------------------------------------------------------------------
    void init_Q(float dt) {
        // IMU 噪声参数 (与 sensors.py 一致)
        // 连续时间 PSD: sigma², 离散化: Q_d = sigma² * dt
        float sigma_gyro    = radians(0.01f);      // rad/s/√Hz 陀螺白噪声
        float sigma_accel   = 0.02f;                // m/s²/√Hz 加计白噪声
        float sigma_bg_walk = radians(1e-4f);       // rad/s 零偏游走 (连续 PSD=sigma²)
        float sigma_ba_walk = 1e-4f;                // m/s² 零偏游走 (连续 PSD=sigma²)

        for (int i = 0; i < 225; ++i) Q[i] = 0.0;

        // position: 无独立噪声 (由速度驱动)
        // velocity: sigma_accel² * dt
        double q_v = static_cast<double>(sigma_accel) * sigma_accel * dt;
        for (int i = 3; i < 6; ++i) Q[i * 15 + i] = q_v;

        // attitude: sigma_gyro² * dt
        double q_theta = static_cast<double>(sigma_gyro) * sigma_gyro * dt;
        for (int i = 6; i < 9; ++i) Q[i * 15 + i] = q_theta;

        // gyro bias: sigma_bg_walk² * dt
        double q_bg = static_cast<double>(sigma_bg_walk) * sigma_bg_walk * dt;
        for (int i = 9; i < 12; ++i) Q[i * 15 + i] = q_bg;

        // accel bias: sigma_ba_walk² * dt
        double q_ba = static_cast<double>(sigma_ba_walk) * sigma_ba_walk * dt;
        for (int i = 12; i < 15; ++i) Q[i * 15 + i] = q_ba;
    }

    // -----------------------------------------------------------------------
    // P 初值 (构造时, 对应 Python __init__)
    // -----------------------------------------------------------------------
    void init_P_construct() {
        for (int i = 0; i < 225; ++i) P[i] = 0.0;
        // position: 1.0 m
        for (int i = 0; i < 3; ++i) P[i * 15 + i] = 1.0;
        // velocity: 0.5 m/s
        for (int i = 3; i < 6; ++i) P[i * 15 + i] = 0.5 * 0.5;
        // attitude: 2 度
        float att_var = radians(2.0f); att_var *= att_var;
        for (int i = 6; i < 9; ++i) P[i * 15 + i] = att_var;
        // gyro bias: 0.1 度/s
        float bg_var = radians(0.1f); bg_var *= bg_var;
        for (int i = 9; i < 12; ++i) P[i * 15 + i] = bg_var;
        // accel bias: 0.05 m/s²
        for (int i = 12; i < 15; ++i) P[i * 15 + i] = 0.05 * 0.05;
    }

    // -----------------------------------------------------------------------
    // P 初值 (reset 时, 对应 Python reset)
    // -----------------------------------------------------------------------
    void init_P_reset() {
        for (int i = 0; i < 225; ++i) P[i] = 0.0;
        // position: 1.0 m
        for (int i = 0; i < 3; ++i) P[i * 15 + i] = 1.0;
        // velocity: 0.5 m/s
        for (int i = 3; i < 6; ++i) P[i * 15 + i] = 0.5 * 0.5;
        // attitude: 1 度 (reset 用 1 度, 构造用 2 度)
        float att_var = radians(1.0f); att_var *= att_var;
        for (int i = 6; i < 9; ++i) P[i * 15 + i] = att_var;
        // gyro bias: 0.1 度/s
        float bg_var = radians(0.1f); bg_var *= bg_var;
        for (int i = 9; i < 12; ++i) P[i * 15 + i] = bg_var;
        // accel bias: 0.01 m/s² (reset 用 0.01, 构造用 0.05)
        for (int i = 12; i < 15; ++i) P[i * 15 + i] = 0.01 * 0.01;
    }

    // -----------------------------------------------------------------------
    // 误差状态注入 (乘性更新)
    // 对应 Python ekf.py _inject_error()
    //   dx = [δp(3), δv(3), δθ(3), δbg(3), δba(3)]
    //   姿态用乘性更新: q_new = q ⊗ δq(δθ), δq = [1, δθ/2] (小角度近似)
    // -----------------------------------------------------------------------
    void inject_error(const double dx[15]) {
        // 位置/速度: 加性
        for (int i = 0; i < 3; ++i) {
            p[i] += static_cast<float>(dx[i]);
            v[i] += static_cast<float>(dx[3 + i]);
        }

        // 姿态: 乘性 (δθ 在 body 系, 右乘)
        float dtheta[3];
        for (int i = 0; i < 3; ++i) dtheta[i] = static_cast<float>(dx[6 + i]);
        float dtheta_norm = std::sqrt(dtheta[0] * dtheta[0] +
                                      dtheta[1] * dtheta[1] +
                                      dtheta[2] * dtheta[2]);

        float dq[4];
        if (dtheta_norm > 1e-12f) {
            // δq = [cos(|δθ|/2), sin(|δθ|/2)*δθ/|δθ|]
            float half = dtheta_norm * 0.5f;
            float s = std::sin(half) / dtheta_norm;
            dq[0] = std::cos(half);
            dq[1] = dtheta[0] * s;
            dq[2] = dtheta[1] * s;
            dq[3] = dtheta[2] * s;
        } else {
            // 小角度近似: δq ≈ [1, δθ/2]
            dq[0] = 1.0f;
            dq[1] = dtheta[0] * 0.5f;
            dq[2] = dtheta[1] * 0.5f;
            dq[3] = dtheta[2] * 0.5f;
        }

        // q_new = q ⊗ δq (body 系误差, 右乘)
        float q_new[4];
        quat_multiply(q, dq, q_new);
        for (int i = 0; i < 4; ++i) q[i] = q_new[i];
        quat_normalize(q);

        // 零偏: 加性
        for (int i = 0; i < 3; ++i) {
            bg[i] += static_cast<float>(dx[9 + i]);
            ba[i] += static_cast<float>(dx[12 + i]);
        }
    }

    // =======================================================================
    // 四元数与 3x3 矩阵辅助函数 (静态, 行主序)
    // 与 Python quaternion_utils.py 严格一致
    // =======================================================================

    static constexpr float PI_F = 3.14159265358979323846f;
    static constexpr float G0   = 9.80665f;

    static float radians(float deg) { return deg * PI_F / 180.0f; }

    // 四元数归一化 (就地)
    static void quat_normalize(float q[4]) {
        float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        if (n > 1e-15f) {
            float inv = 1.0f / n;
            for (int i = 0; i < 4; ++i) q[i] *= inv;
        } else {
            q[0] = 1.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 0.0f;
        }
    }

    // Hamilton 乘法 q_out = q1 ⊗ q2, q=[w,x,y,z]
    static void quat_multiply(const float q1[4], const float q2[4], float q_out[4]) {
        q_out[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
        q_out[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
        q_out[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
        q_out[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    }

    // 四元数 -> 旋转矩阵 C_bn (3×3, 行主序), v_n = C @ v_b
    static void quat_to_rotmat(const float q[4], float C[9]) {
        float w = q[0], x = q[1], y = q[2], z = q[3];
        C[0] = 1.0f - 2.0f * (y * y + z * z);
        C[1] = 2.0f * (x * y - w * z);
        C[2] = 2.0f * (x * z + w * y);
        C[3] = 2.0f * (x * y + w * z);
        C[4] = 1.0f - 2.0f * (x * x + z * z);
        C[5] = 2.0f * (y * z - w * x);
        C[6] = 2.0f * (x * z - w * y);
        C[7] = 2.0f * (y * z + w * x);
        C[8] = 1.0f - 2.0f * (x * x + y * y);
    }

    // 四元数运动学 q_dot = 0.5 * q ⊗ [0, omega]
    static void quat_kinematics(const float q[4], const float omega[3], float q_dot[4]) {
        float omega_q[4] = {0.0f, omega[0], omega[1], omega[2]};
        quat_multiply(q, omega_q, q_dot);
        for (int i = 0; i < 4; ++i) q_dot[i] *= 0.5f;
    }

    // 反对称矩阵 [v]× (3×3, 行主序)
    static void skew(const float v[3], float S[9]) {
        S[0] = 0.0f;  S[1] = -v[2]; S[2] = v[1];
        S[3] = v[2];  S[4] = 0.0f;  S[5] = -v[0];
        S[6] = -v[1]; S[7] = v[0];  S[8] = 0.0f;
    }

    // 3×3 矩阵乘法 C = A * B (行主序)
    static void mat3_mul(const float A[9], const float B[9], float C[9]) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    sum += A[i * 3 + k] * B[k * 3 + j];
                }
                C[i * 3 + j] = sum;
            }
        }
    }

    // 3×3 矩阵 * 3×1 向量: y = A * x
    static void mat3_vec_mul(const float A[9], const float x[3], float y[3]) {
        for (int i = 0; i < 3; ++i) {
            y[i] = A[i * 3 + 0] * x[0] + A[i * 3 + 1] * x[1] + A[i * 3 + 2] * x[2];
        }
    }
};

}  // namespace falcon9
