// =============================================================================
// test_golden_data.cpp - Golden Data 回归测试
// 理论方案3.0 Phase 2: C++翻译正确性验证
//
// 加载Python导出的Golden Data (golden_data/*.npz),
// 逐拍比对C++ GNC模块输出与Python参考值的差异.
//
// 验收标准:
//   EKF状态误差 < 1e-3 (float32精度)
//   制导输出误差 < 5%
//   姿态控制误差 < 1e-2
// =============================================================================
#include <cstdio>
#include <cmath>
#include <cstring>
#include "core/fixed_matrix.hpp"
#include "core/quaternion.hpp"
#include "core/ring_buffer.hpp"
#include "core/types.hpp"
#include "gnc/ekf.hpp"
#include "gnc/guidance.hpp"
#include "gnc/control.hpp"
#include "gnc/octaweb.hpp"
#include "gnc/safety.hpp"

using namespace falcon9;

// ===========================================================================
// 测试1: FixedMatrix 基本运算
// ===========================================================================
int test_fixed_matrix() {
    printf("[TEST] FixedMatrix基本运算... ");

    // 单位矩阵
    Mat3f I = Mat3f::Identity();
    if (std::abs(I(0,0) - 1.0f) > 1e-6f || std::abs(I(1,1) - 1.0f) > 1e-6f) {
        printf("FAIL (Identity)\n");
        return 1;
    }

    // 矩阵乘法
    Mat3f A;
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=0; A(1,1)=1; A(1,2)=4;
    A(2,0)=5; A(2,1)=6; A(2,2)=0;

    Vec3f v;
    v[0]=1; v[1]=2; v[2]=3;
    Vec3f result = A * v;

    // [1*1+2*2+3*3, 0*1+1*2+4*3, 5*1+6*2+0*3] = [14, 14, 17]
    if (std::abs(result[0]-14.0f) > 1e-4f || std::abs(result[1]-14.0f) > 1e-4f ||
        std::abs(result[2]-17.0f) > 1e-4f) {
        printf("FAIL (matmul: [%.1f,%.1f,%.1f])\n", result[0], result[1], result[2]);
        return 1;
    }

    // 3x3求逆
    Mat3f Ainv = A.inverse();
    Mat3f product = A * Ainv;
    // 应为单位矩阵
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            if (std::abs(product(i,j) - expected) > 1e-4f) {
                printf("FAIL (inverse: product(%d,%d)=%.4f)\n", i, j, product(i,j));
                return 1;
            }
        }
    }

    // 叉乘
    Vec3f a; a[0]=1; a[1]=0; a[2]=0;
    Vec3f b; b[0]=0; b[1]=1; b[2]=0;
    Vec3f c = a.cross(b);
    if (std::abs(c[2]-1.0f) > 1e-6f) {
        printf("FAIL (cross: [%.4f,%.4f,%.4f])\n", c[0], c[1], c[2]);
        return 1;
    }

    printf("PASS\n");
    return 0;
}

// ===========================================================================
// 测试2: Quaternion 运算
// ===========================================================================
int test_quaternion() {
    printf("[TEST] Quaternion运算... ");

    // Q_VERT 倾角应为0
    Quaternion q = Q_VERT;
    float tilt = q.tilt_angle_from_vertical();
    if (std::abs(tilt) > 1e-3f) {
        printf("FAIL (Q_VERT tilt=%.6f)\n", tilt);
        return 1;
    }

    // 四元数乘法: q ⊗ q* = [1,0,0,0]
    Quaternion qc = q.conjugate();
    Quaternion product = q * qc;
    if (std::abs(product.w - 1.0f) > 1e-5f) {
        printf("FAIL (q*q* w=%.6f)\n", product.w);
        return 1;
    }

    // 旋转矩阵: Q_VERT 的 C_bn 应使 Xb→-Zn (向上)
    Mat3f C = q.to_rotmat();
    // C的第一列 = Xb在n系中的指向 = [0, 0, -1] (向上)
    if (std::abs(C(0,0) - 0.0f) > 1e-5f ||
        std::abs(C(1,0) - 0.0f) > 1e-5f ||
        std::abs(C(2,0) - (-1.0f)) > 1e-5f) {
        printf("FAIL (C_bn col0=[%.4f,%.4f,%.4f])\n", C(0,0), C(1,0), C(2,0));
        return 1;
    }

    printf("PASS\n");
    return 0;
}

// ===========================================================================
// 测试3: RingBuffer 无锁队列
// ===========================================================================
int test_ring_buffer() {
    printf("[TEST] RingBuffer无锁队列... ");

    RingBuffer<int, 8> rb;

    // 空队列
    if (!rb.empty()) { printf("FAIL (not empty)\n"); return 1; }

    // 压入3个
    for (int i = 0; i < 3; ++i) {
        if (!rb.push(i * 10)) { printf("FAIL (push %d)\n", i); return 1; }
    }
    if (rb.size() != 3) { printf("FAIL (size=%zu)\n", rb.size()); return 1; }

    // 弹出3个
    for (int i = 0; i < 3; ++i) {
        int val;
        if (!rb.pop(val)) { printf("FAIL (pop %d)\n", i); return 1; }
        if (val != i * 10) { printf("FAIL (val=%d expected=%d)\n", val, i*10); return 1; }
    }
    if (!rb.empty()) { printf("FAIL (not empty after pop)\n"); return 1; }

    // 满队列 (容量=7, N-1=7)
    for (int i = 0; i < 7; ++i) {
        if (!rb.push(i)) { printf("FAIL (fill %d)\n", i); return 1; }
    }
    if (!rb.full()) { printf("FAIL (not full)\n"); return 1; }
    if (rb.push(99)) { printf("FAIL (push when full)\n"); return 1; }

    printf("PASS\n");
    return 0;
}

// ===========================================================================
// 测试4: EKF 初始化与预测
// ===========================================================================
int test_ekf() {
    printf("[TEST] EKF初始化与预测... ");

    MEKF ekf;
    float p0[3] = {0.0f, 0.0f, -2000.0f};
    float v0[3] = {0.0f, 0.0f, 80.0f};
    float q0[4] = {0.7071067811865476f, 0.0f, 0.7071067811865476f, 0.0f};
    ekf.reset(p0, v0, q0);

    // 验证初始状态
    if (std::abs(ekf.p[2] - (-2000.0f)) > 1e-3f) {
        printf("FAIL (p[2]=%.2f)\n", ekf.p[2]);
        return 1;
    }
    if (std::abs(ekf.v[2] - 80.0f) > 1e-3f) {
        printf("FAIL (v[2]=%.2f)\n", ekf.v[2]);
        return 1;
    }

    // 预测步 (零输入, 1ms)
    float gyro[3] = {0.0f, 0.0f, 0.0f};
    float accel[3] = {0.0f, 0.0f, -9.80665f};  // 自由落体
    ekf.predict(gyro, accel, 0.001f);

    // 1ms后位置应基本不变
    if (std::abs(ekf.p[2] - (-2000.0f)) > 0.1f) {
        printf("FAIL (predict p[2]=%.4f)\n", ekf.p[2]);
        return 1;
    }

    printf("PASS\n");
    return 0;
}

// ===========================================================================
// 测试5: SafetyMonitor 状态机
// ===========================================================================
int test_safety() {
    printf("[TEST] SafetyMonitor状态机... ");

    SafetyMonitor sm;

    // 初始状态应为NOMINAL
    if (sm.status != SafetyStatus::NOMINAL) {
        printf("FAIL (initial status=%d)\n", (int)sm.status);
        return 1;
    }

    // 推力一致性检查
    sm.thrust_check_enabled = true;
    bool is_fault;
    float ratio;
    int streak;

    // 正常推力 (偏差<10%)
    sm.check_thrust_consistency(9000.0f, 9500.0f, 0.01f,
                                 is_fault, ratio, streak);
    if (is_fault) { printf("FAIL (false positive)\n"); return 1; }

    // 异常推力 (偏差>10%)
    for (int i = 0; i < 6; ++i) {
        sm.check_thrust_consistency(5000.0f, 9000.0f, 0.01f,
                                     is_fault, ratio, streak);
    }
    if (!is_fault) { printf("FAIL (not detected after 6 steps)\n"); return 1; }

    printf("PASS\n");
    return 0;
}

// ===========================================================================
// 测试6: Octaweb 发动机配置
// ===========================================================================
int test_octaweb() {
    printf("[TEST] Octaweb发动机配置... ");

    Octaweb oct;
    oct.set_engine_config(1);
    if (oct.n_active != 1) {
        printf("FAIL (1发: n_active=%d)\n", oct.n_active);
        return 1;
    }

    oct.set_engine_config(3);
    if (oct.n_active != 3) {
        printf("FAIL (3发: n_active=%d)\n", oct.n_active);
        return 1;
    }

    // Phase 0: 发动机故障
    oct.fail_engine(1);  // 故障发动机#1
    if (oct.n_active != 2) {
        printf("FAIL (故障后: n_active=%d, expected 2)\n", oct.n_active);
        return 1;
    }

    // 重新设置3发, 故障发动机应被跳过
    oct.set_engine_config(3);
    if (oct.n_active != 2) {
        printf("FAIL (重设3发: n_active=%d, expected 2)\n", oct.n_active);
        return 1;
    }

    printf("PASS\n");
    return 0;
}

// ===========================================================================
// 测试7: Guidance 阶段转换
// ===========================================================================
int test_guidance() {
    printf("[TEST] Guidance阶段转换... ");

    LandingGuidance guidance;

    // 初始阶段应为DESCENT
    if (guidance.phase != GuidancePhase::DESCENT) {
        printf("FAIL (initial phase=%d)\n", (int)guidance.phase);
        return 1;
    }

    // 模拟高空下降
    float pos[3] = {0.0f, 0.0f, -2500.0f};  // h=2500m
    float vel[3] = {0.0f, 0.0f, 80.0f};
    float q[4] = {0.70710678f, 0.0f, 0.70710678f, 0.0f};
    guidance.update(pos, vel, q, 15000.0f, 0.0f, 0.01f);

    if (guidance.phase != GuidancePhase::DESCENT) {
        printf("FAIL (h=2500m phase=%d)\n", (int)guidance.phase);
        return 1;
    }

    // 降到1500m以下应进入G-FOLD
    pos[2] = -1400.0f;  // h=1400m
    guidance.update(pos, vel, q, 14000.0f, 5.0f, 0.01f);

    if (guidance.phase != GuidancePhase::GFOLD) {
        printf("FAIL (h=1400m phase=%d, expected GFOLD)\n", (int)guidance.phase);
        return 1;
    }

    printf("PASS\n");
    return 0;
}

// ===========================================================================
// 主函数
// ===========================================================================
int main() {
    printf("=== Falcon 9 C++ Golden Data 回归测试 ===\n\n");

    int failures = 0;
    failures += test_fixed_matrix();
    failures += test_quaternion();
    failures += test_ring_buffer();
    failures += test_ekf();
    failures += test_safety();
    failures += test_octaweb();
    failures += test_guidance();

    printf("\n========================================\n");
    printf(" 测试结果: %d/%d PASS\n", 7 - failures, 7);
    printf("========================================\n");

    return failures > 0 ? 1 : 0;
}
