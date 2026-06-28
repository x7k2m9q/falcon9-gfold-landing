// =============================================================================
// sensor_guard.hpp - 传感器数据净化层 (工程硬化阶段一)
//
// 理论方案4.0 阶段一: 工程底座航天级硬化
//   在 SensorData 进入 EKF/控制律之前, 执行:
//     1. NaN/Inf 检测与替换 (用上一拍有效值回退)
//     2. 物理量程钳位 (防止超量程值破坏数值稳定)
//     3. 时间戳合理性检查 (防止倒退/跳跃导致燃料计算异常)
//
// 设计原则:
//   - 零动态内存 (嵌入式约束)
//   - 单拍延迟回退 (last_good 缓存, 不做预测外推)
//   - 保守钳位 (宁可信息损失, 不可数值发散)
//   - 故障计数器: 连续 N 拍异常 → 上报 SAFE_MODE
// =============================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include "../core/types.hpp"

namespace falcon9 {

// ---------------------------------------------------------------------------
// 物理量程限制 (对应真实传感器规格)
// ---------------------------------------------------------------------------
namespace sensor_limits {
    // 陀螺仪: Falcon 9 栅格舵工作段 <0.5 rad/s, 量程上限 10 rad/s
    constexpr float GYRO_MAX       = 10.0f;       // [rad/s]
    constexpr float GYRO_MIN       = -10.0f;

    // 加速度计: 回收段 <15g, 量程上限 50g
    constexpr float ACCEL_MAX      = 500.0f;      // [m/s²] (~50g)
    constexpr float ACCEL_MIN      = -500.0f;

    // GPS 位置: 地球半径量级, 超过 1e5 m 视为异常
    constexpr float GPS_POS_MAX    = 1.0e5f;      // [m]
    constexpr float GPS_POS_MIN    = -1.0e5f;

    // GPS 速度: 回收段 <300 m/s, 量程上限 1000 m/s
    constexpr float GPS_VEL_MAX    = 1000.0f;     // [m/s]
    constexpr float GPS_VEL_MIN    = -1000.0f;

    // 雷达高度: 0 ~ 5000 m
    constexpr float RADAR_ALT_MAX  = 5000.0f;     // [m]
    constexpr float RADAR_ALT_MIN  = 0.0f;

    // 时间戳: 最大合理间隔 1 秒 (1000000 us)
    constexpr uint32_t DT_MAX_US   = 1000000;     // [us]
    constexpr uint32_t DT_MIN_US   = 100;         // [us] (10kHz 上限)
}

// ===========================================================================
// SensorGuard - 传感器数据净化器
//
// 使用方法:
//   SensorGuard guard;
//   guard.sanitize(sensor);  // 就地净化, 返回 true 表示数据可用
//
// 内部缓存上一拍有效数据, 用于 NaN 回退.
// 连续异常计数超过阈值时, 设置 data_invalid 标志.
// ===========================================================================
class SensorGuard {
public:
    // 连续异常计数阈值 (超过此值 → 数据不可信)
    static constexpr int FAULT_THRESHOLD = 10;  // 10 拍 = 0.1s @100Hz

    // 缓存的上一拍有效数据
    SensorData last_good;

    // 连续异常计数
    int fault_count;

    // 数据完全不可用标志 (连续异常超过阈值)
    bool data_invalid;

    // 诊断: 最近替换的字段 (位图)
    //   bit 0: gyro
    //   bit 1: accel
    //   bit 2: gps_pos
    //   bit 3: gps_vel
    //   bit 4: radar_alt
    //   bit 5: timestamp
    uint8_t replaced_mask;

    SensorGuard() : fault_count(0), data_invalid(false), replaced_mask(0) {
        last_good.reset();
        // 初始化为合理的默认值 (不是零, 避免首拍异常)
        last_good.accel[2] = -9.80665f;  // 重力 (NED Z 向下)
        last_good.gps_pos[2] = -1000.0f; // 1000m 高度
        last_good.radar_alt = 1000.0f;
        last_good.gps_valid = false;     // 首拍不信任 GPS
        last_good.radar_valid = false;
        last_good.timestamp_us = 0;
    }

    // -----------------------------------------------------------------------
    // 检测 float 是否为 NaN 或 Inf
    // -----------------------------------------------------------------------
    static inline bool is_bad(float v) {
        return std::isnan(v) || std::isinf(v);
    }

    // -----------------------------------------------------------------------
    // 钳位到 [lo, hi], 若 bad 则返回 fallback
    // -----------------------------------------------------------------------
    static inline float clamp_or(float v, float lo, float hi, float fallback) {
        if (is_bad(v)) return fallback;
        if (v > hi) return hi;
        if (v < lo) return lo;
        return v;
    }

    // -----------------------------------------------------------------------
    // 就地净化 SensorData
    //   返回 true: 数据可用 (可能经过钳位/回退)
    //   返回 false: 数据完全不可信 (连续异常超过阈值)
    // -----------------------------------------------------------------------
    bool sanitize(SensorData& s) {
        replaced_mask = 0;
        bool any_replaced = false;

        // === 1. 陀螺仪 ===
        for (int i = 0; i < 3; ++i) {
            if (is_bad(s.gyro[i])) {
                s.gyro[i] = last_good.gyro[i];
                replaced_mask |= (1 << 0);
                any_replaced = true;
            } else if (s.gyro[i] > sensor_limits::GYRO_MAX) {
                s.gyro[i] = sensor_limits::GYRO_MAX;
                replaced_mask |= (1 << 0);
                any_replaced = true;
            } else if (s.gyro[i] < sensor_limits::GYRO_MIN) {
                s.gyro[i] = sensor_limits::GYRO_MIN;
                replaced_mask |= (1 << 0);
                any_replaced = true;
            }
        }

        // === 2. 加速度计 ===
        for (int i = 0; i < 3; ++i) {
            if (is_bad(s.accel[i])) {
                s.accel[i] = last_good.accel[i];
                replaced_mask |= (1 << 1);
                any_replaced = true;
            } else if (s.accel[i] > sensor_limits::ACCEL_MAX) {
                s.accel[i] = sensor_limits::ACCEL_MAX;
                replaced_mask |= (1 << 1);
                any_replaced = true;
            } else if (s.accel[i] < sensor_limits::ACCEL_MIN) {
                s.accel[i] = sensor_limits::ACCEL_MIN;
                replaced_mask |= (1 << 1);
                any_replaced = true;
            }
        }

        // === 3. GPS 位置 ===
        bool gps_pos_bad = false;
        for (int i = 0; i < 3; ++i) {
            if (is_bad(s.gps_pos[i])) {
                gps_pos_bad = true;
                break;
            }
        }
        if (gps_pos_bad) {
            // GPS 位置整体回退 (不做分量级回退, 保持空间一致性)
            for (int i = 0; i < 3; ++i) s.gps_pos[i] = last_good.gps_pos[i];
            s.gps_valid = false;  // 标记 GPS 无效
            replaced_mask |= (1 << 2);
            any_replaced = true;
        } else {
            // 钳位
            for (int i = 0; i < 3; ++i) {
                if (s.gps_pos[i] > sensor_limits::GPS_POS_MAX) {
                    s.gps_pos[i] = sensor_limits::GPS_POS_MAX;
                    replaced_mask |= (1 << 2);
                    any_replaced = true;
                } else if (s.gps_pos[i] < sensor_limits::GPS_POS_MIN) {
                    s.gps_pos[i] = sensor_limits::GPS_POS_MIN;
                    replaced_mask |= (1 << 2);
                    any_replaced = true;
                }
            }
        }

        // === 4. GPS 速度 ===
        bool gps_vel_bad = false;
        for (int i = 0; i < 3; ++i) {
            if (is_bad(s.gps_vel[i])) {
                gps_vel_bad = true;
                break;
            }
        }
        if (gps_vel_bad) {
            for (int i = 0; i < 3; ++i) s.gps_vel[i] = last_good.gps_vel[i];
            s.gps_valid = false;
            replaced_mask |= (1 << 3);
            any_replaced = true;
        } else {
            for (int i = 0; i < 3; ++i) {
                if (s.gps_vel[i] > sensor_limits::GPS_VEL_MAX) {
                    s.gps_vel[i] = sensor_limits::GPS_VEL_MAX;
                    replaced_mask |= (1 << 3);
                    any_replaced = true;
                } else if (s.gps_vel[i] < sensor_limits::GPS_VEL_MIN) {
                    s.gps_vel[i] = sensor_limits::GPS_VEL_MIN;
                    replaced_mask |= (1 << 3);
                    any_replaced = true;
                }
            }
        }

        // === 5. 雷达高度 ===
        if (is_bad(s.radar_alt)) {
            s.radar_alt = last_good.radar_alt;
            s.radar_valid = false;
            replaced_mask |= (1 << 4);
            any_replaced = true;
        } else if (s.radar_alt > sensor_limits::RADAR_ALT_MAX) {
            s.radar_alt = sensor_limits::RADAR_ALT_MAX;
            replaced_mask |= (1 << 4);
            any_replaced = true;
        } else if (s.radar_alt < sensor_limits::RADAR_ALT_MIN) {
            // 负高度无物理意义, 标记雷达无效 (不用 0, 避免 EKF 误判着陆)
            s.radar_valid = false;
            replaced_mask |= (1 << 4);
            any_replaced = true;
        }

        // === 6. 时间戳合理性 ===
        // 时间戳为零或倒退 (相对 last_good) → 用 last_good + DT_MIN_US 推进
        if (s.timestamp_us == 0 ||
            (last_good.timestamp_us > 0 &&
             s.timestamp_us <= last_good.timestamp_us)) {
            // 时间戳异常: 用上一拍 + 最小步长推进, 保证单调递增
            s.timestamp_us = last_good.timestamp_us + sensor_limits::DT_MIN_US;
            replaced_mask |= (1 << 5);
            any_replaced = true;
        }
        // 时间戳跳跃过大 (>1秒) → 钳位
        if (last_good.timestamp_us > 0) {
            uint32_t dt = s.timestamp_us - last_good.timestamp_us;
            if (dt > sensor_limits::DT_MAX_US) {
                s.timestamp_us = last_good.timestamp_us + sensor_limits::DT_MAX_US;
                replaced_mask |= (1 << 5);
                any_replaced = true;
            }
        }

        // === 7. 更新故障计数 ===
        if (any_replaced) {
            fault_count++;
            if (fault_count > FAULT_THRESHOLD) {
                data_invalid = true;
            }
        } else {
            // 数据正常, 更新 last_good, 清除故障计数
            last_good = s;
            fault_count = 0;
            data_invalid = false;
        }

        // 注意: 即使有替换, 也更新 last_good (用净化后的值)
        // 这样下一拍的回退基准是净化后的合理值, 不是原始的异常值
        if (any_replaced && !data_invalid) {
            last_good = s;
        }

        return !data_invalid;
    }

    // -----------------------------------------------------------------------
    // 诊断: 返回最近替换的字段描述
    // -----------------------------------------------------------------------
    void get_replaced_desc(char* buf, int buf_size) const {
        if (replaced_mask == 0) {
            snprintf(buf, buf_size, "none");
            return;
        }
        int offset = 0;
        const char* names[] = {"gyro", "accel", "gps_pos", "gps_vel", "radar", "ts"};
        bool first = true;
        for (int i = 0; i < 6; ++i) {
            if (replaced_mask & (1 << i)) {
                offset += snprintf(buf + offset, buf_size - offset,
                                   "%s%s", first ? "" : "|", names[i]);
                first = false;
            }
        }
        if (offset == 0) {
            snprintf(buf, buf_size, "none");
        }
    }
};

}  // namespace falcon9
