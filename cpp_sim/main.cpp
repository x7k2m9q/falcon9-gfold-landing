// =============================================================================
// main.cpp - 猎鹰9号 C++ 飞控仿真主程序
// 理论方案3.0 Phase 3: FreeRTOS软硬件协同验证
//
// 架构:
//   Python物理引擎 (dynamics.py) ←UDP→ C++飞控 (4个FreeRTOS任务)
//
// 启动流程:
//   1. 初始化UDP (端口28015接收, 28016发送)
//   2. 创建FlightComputer (EKF/Guidance/Control/Octaweb/Safety)
//   3. 启动4个任务线程 (IMU 1000Hz / Control 100Hz / Safety 10Hz / Guidance 1Hz)
//   4. 主循环: UDP接收传感器数据 → 写入FlightComputer → 读取控制输出 → UDP发送
//   5. 着陆或超时后停止, 输出统计
//
// 验收标准 (理论方案3.0 Phase 3):
//   - Valgrind: 无内存泄漏
//   - Jitter: <1ms
//   - 精度: 与Python Golden Data误差 <20%
//   - 6类极限测试: 全部通过
// =============================================================================
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include "core/fixed_matrix.hpp"
#include "core/quaternion.hpp"
#include "core/ring_buffer.hpp"
#include "core/types.hpp"
#include "gnc/ekf.hpp"
#include "gnc/guidance.hpp"
#include "gnc/control.hpp"
#include "gnc/octaweb.hpp"
#include "gnc/safety.hpp"
#include "os/freertos_sim.hpp"
#include "hal_sim/udp_interface.hpp"

using namespace falcon9;

// =============================================================================
// 主函数
// =============================================================================
int main(int argc, char* argv[]) {
    // 工程硬化: 禁用 stdout 缓冲, 确保崩溃前输出可见
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    printf("=== Falcon 9 C++ 飞控仿真 ===\n");
    printf("理论方案3.0 Phase 3: FreeRTOS软硬件协同验证\n\n");

    // --- 解析参数 ---
    int recv_port = 28015;
    int send_port = 28016;
    const char* dest_ip = "127.0.0.1";
    double max_sim_time = 200.0;  // 最大仿真时间 [s]

    // 安全字符串转整数 (校验合法性)
    auto parse_port = [](const char* s, int* out) -> bool {
        char* endp = nullptr;
        long v = strtol(s, &endp, 10);
        if (endp == s || *endp != '\0' || v < 1 || v > 65535) return false;
        *out = static_cast<int>(v);
        return true;
    };
    auto parse_double = [](const char* s, double* out) -> bool {
        char* endp = nullptr;
        double v = strtod(s, &endp);
        if (endp == s || *endp != '\0') return false;
        *out = v;
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--recv") == 0 && i + 1 < argc) {
            if (!parse_port(argv[++i], &recv_port)) {
                printf("[FATAL] 非法端口: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--send") == 0 && i + 1 < argc) {
            if (!parse_port(argv[++i], &send_port)) {
                printf("[FATAL] 非法端口: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--ip") == 0 && i + 1 < argc)
            dest_ip = argv[++i];
        else if (strcmp(argv[i], "--maxtime") == 0 && i + 1 < argc) {
            if (!parse_double(argv[++i], &max_sim_time) || max_sim_time <= 0) {
                printf("[FATAL] 非法时间: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("用法: falcon9_sim [选项]\n");
            printf("  --recv <port>     接收端口 (默认28015)\n");
            printf("  --send <port>     发送端口 (默认28016)\n");
            printf("  --ip <addr>       目标IP (默认127.0.0.1)\n");
            printf("  --maxtime <sec>   最大仿真时间 (默认200)\n");
            return 0;
        }
    }

    printf("配置: recv=%d send=%d ip=%s maxtime=%.0fs\n\n",
           recv_port, send_port, dest_ip, max_sim_time);

    // --- 初始化UDP ---
    if (!udp_init()) {
        printf("[FATAL] UDP初始化失败\n");
        return 1;
    }

    UdpReceiver receiver;
    UdpSender   sender;

    if (!receiver.init(recv_port)) {
        printf("[FATAL] UDP接收器初始化失败\n");
        udp_cleanup();
        return 1;
    }
    if (!sender.init(dest_ip, send_port)) {
        printf("[FATAL] UDP发送器初始化失败\n");
        receiver.close();
        udp_cleanup();
        return 1;
    }

    // --- 创建飞控计算机 ---
    FlightComputer fc;
    std::atomic<bool> sensor_ready{false};

    // 初始化EKF (与Python一致: p0=[0,0,-2000], v0=[0,0,80], q=Q_VERT)
    float p0[3] = {0.0f, 0.0f, -2000.0f};
    float v0[3] = {0.0f, 0.0f, 80.0f};
    float q0[4] = {0.7071067811865476f, 0.0f, 0.7071067811865476f, 0.0f};
    fc.ekf.reset(p0, v0, q0);

    printf("[INIT] EKF初始化: p=[%.0f,%.0f,%.0f] v=[%.0f,%.0f,%.0f]\n",
           p0[0], p0[1], p0[2], v0[0], v0[1], v0[2]);
    printf("[INIT] Octaweb: %d发模式\n", fc.octaweb.n_active);
    printf("[INIT] Guidance: 阶段=DESCENT\n\n");

    // --- 启动4个任务线程 ---
    printf("[BOOT] 启动FreeRTOS任务模拟...\n");
    printf("  Task_IMU      1000Hz (优先级5)\n");
    printf("  Task_Control   100Hz (优先级4)\n");
    printf("  Task_Safety     10Hz (优先级3)\n");
    printf("  Task_Guidance    1Hz (优先级2)\n\n");

    auto threads = launch_flight_computer(fc, sensor_ready);

    // --- 主循环: UDP中继 ---
    printf("[RUN] 主循环启动, 等待Python物理引擎连接...\n");

    auto sim_start = Clock::now();
    int packets_rx = 0;
    int packets_tx = 0;
    int packets_lost = 0;
    uint32_t last_timestamp = 0;

    while (fc.running.load(std::memory_order_relaxed)) {
        // 检查超时
        double elapsed = elapsed_ms(sim_start, Clock::now()) / 1000.0;
        if (elapsed > max_sim_time) {
            printf("[TIMEOUT] 仿真超时 %.1fs\n", elapsed);
            break;
        }

        // 检查着陆
        if (fc.landed.load(std::memory_order_relaxed)) {
            printf("[LANDED] 着陆完成!\n");
            break;
        }

        // 接收传感器数据
        SensorData sensor;
        if (receiver.receive(sensor)) {
            packets_rx++;

            // 时间戳连续性检查
            if (last_timestamp > 0 && sensor.timestamp_us < last_timestamp) {
                packets_lost++;
            }
            last_timestamp = sensor.timestamp_us;

            // 直接处理传感器数据 (EKF predict + update)
            // 注: 原架构通过 sensor_ready 标志由 Task_IMU 1000Hz 处理,
            //     但 Windows 定时器分辨率 15.6ms 导致 Task_IMU 实际仅 ~64Hz,
            //     EKF 严重滞后. 现改为在主循环 100Hz 直接处理, 匹配 dt=0.01.

            // === 工程硬化阶段一: 传感器数据净化 ===
            // 在进入 EKF 之前, 检测 NaN/Inf, 钳位超量程值, 回退到上一拍有效值
            bool data_ok = fc.sensor_guard.sanitize(sensor);
            if (!data_ok) {
                printf("[GUARD] 传感器数据持续异常, 进入 SAFE_MODE\n");
            }
            if (fc.sensor_guard.replaced_mask != 0 && (packets_rx % 100 == 0)) {
                char desc[64];
                fc.sensor_guard.get_replaced_desc(desc, sizeof(desc));
                printf("[GUARD] 替换字段: %s (fault_count=%d)\n",
                       desc, fc.sensor_guard.fault_count);
            }

            fc.latest_sensor = sensor;
            process_sensor_data(fc, sensor);

            // 更新仿真时间
            fc.sim_time.store(static_cast<float>(sensor.timestamp_us) / 1e6f,
                              std::memory_order_relaxed);

            // 执行控制逻辑 (100Hz, 与 EKF 同步)
            process_control(fc);

            // 发送控制输出 (等待至少一个控制周期)
            if (fc.control_steps.load(std::memory_order_relaxed) > 0) {
                if (sender.send(fc.latest_control)) {
                    packets_tx++;
                }
            }

            // 更新燃料 (C++端本地估算, 对应 Python dyn.step 中的燃料消耗)
            // mdot = T / (Isp * g0), T = throttle * T_max_single * n_engines
            if (fc.control_steps.load(std::memory_order_relaxed) > 0) {
                float throttle = fc.latest_control.throttle;
                int n_engines = (fc.latest_control.n_engines == EngineConfig::TRIPLE) ? 3 : 1;
                float T_max_single = rocket_params::thrust_at_alt(-fc.latest_state.p[2]);
                float T_total = throttle * T_max_single * static_cast<float>(n_engines);
                float isp = rocket_params::isp_at_alt(-fc.latest_state.p[2]);
                float mdot = T_total / (isp * rocket_params::G0);
                // 估算时间步长 (从时间戳差值)
                static uint32_t prev_timestamp = 0;
                float dt = 0.01f;  // 默认 10ms
                if (prev_timestamp > 0) {
                    uint32_t dt_us = sensor.timestamp_us - prev_timestamp;
                    if (dt_us > 0 && dt_us < 1000000) {
                        dt = static_cast<float>(dt_us) / 1e6f;
                    }
                }
                prev_timestamp = sensor.timestamp_us;
                float fuel_new = fc.fuel_mass.load(std::memory_order_relaxed) - mdot * dt;
                if (fuel_new < 0.0f) fuel_new = 0.0f;
                fc.fuel_mass.store(fuel_new, std::memory_order_relaxed);
            }
        }
    }

    // --- 停止任务 ---
    fc.stop();
    printf("\n[STOP] 等待任务线程退出...\n");
    join_flight_computer(threads);

    // --- 统计报告 ---
    auto sim_end = Clock::now();
    double total_time = elapsed_ms(sim_start, sim_end) / 1000.0;

    printf("\n");
    printf("========================================\n");
    printf(" Phase 3 仿真统计报告\n");
    printf("========================================\n");
    printf(" 总仿真时间:     %.2fs\n", total_time);
    printf(" 仿真时钟:       %.2fs\n", fc.sim_time.load());
    printf(" ---------------------------------------\n");
    printf(" Task_IMU步数:   %d  (jitter_max=%.3fms)\n",
           fc.imu_steps.load(), fc.imu_jitter_max.load());
    printf(" Task_Control步: %d  (jitter_max=%.3fms)\n",
           fc.control_steps.load(), fc.control_jitter_max.load());
    printf(" Task_Safety步:  %d\n", fc.safety_steps.load());
    printf(" Task_Guidance步:%d\n", fc.guidance_steps.load());
    printf(" ---------------------------------------\n");
    printf(" UDP接收:        %d 包\n", packets_rx);
    printf(" UDP发送:        %d 包\n", packets_tx);
    printf(" UDP丢包:        %d\n", packets_lost);
    printf(" ---------------------------------------\n");
    printf(" 着陆状态:       %s\n",
           fc.landed.load() ? "成功" : "未完成");
    printf(" 最终安全状态:   %d\n",
           static_cast<int>(fc.latest_safety));
    printf(" 最终发动机:     %d发\n", fc.octaweb.n_active);
    printf("========================================\n");

    // --- 验收检查 ---
    printf("\n验收检查:\n");
    bool jitter_ok = (fc.imu_jitter_max.load() < 1.0 &&
                      fc.control_jitter_max.load() < 1.0);
    printf("  Jitter <1ms:    %s (IMU=%.3fms Ctrl=%.3fms)\n",
           jitter_ok ? "PASS" : "FAIL",
           fc.imu_jitter_max.load(), fc.control_jitter_max.load());
    printf("  着陆成功:       %s\n",
           fc.landed.load() ? "PASS" : "FAIL");

    // --- 清理 ---
    receiver.close();
    sender.close();
    udp_cleanup();

    printf("\n[DONE] 仿真结束\n");
    return fc.landed.load() ? 0 : 1;
}
