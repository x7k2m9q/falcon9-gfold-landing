"""最终100次蒙特卡洛 - 分批增量版本.

每10个种子保存一次中间结果到mc100_progress.npy, 避免长时间运行被中断丢失数据.
支持断点续跑: 如果检测到已有进度文件, 从上次断点继续.
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from e6_hardening import run_e6

PROGRESS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'mc100_progress.pkl')
RESULT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'mc100_results.txt')


def load_progress():
    """加载已有进度. 返回 (results, next_idx)."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'rb') as f:
            data = pickle.load(f)
        print("恢复进度: 已完成 %d/%d" % (len(data['results']), data['n_total']))
        return data['results'], data['next_idx'], data['seeds'], data['n_total']
    return [], 0, None, 100


def save_progress(results, next_idx, seeds, n_total):
    """保存进度."""
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump({'results': results, 'next_idx': next_idx,
                     'seeds': seeds, 'n_total': n_total}, f)


def run_batch(seeds, start_idx, batch_size=10):
    """运行一批种子. 返回结果列表."""
    results = []
    end_idx = min(start_idx + batch_size, len(seeds))
    for idx in range(start_idx, end_idx):
        seed = seeds[idx]
        t0 = time.time()
        try:
            success, m = run_e6(seed=seed, verbose=False)
        except Exception as e:
            print("[ERR] seed=%d 异常: %s" % (seed, str(e)))
            success = False
            m = {'h_final': 9999, 'vz_final': 9999, 'tilt_final': 9999,
                 'pos_err_final': 9999, 't_final': 9999, 'fuel_final': 0,
                 'pos_err_ekf': 9999, 'vel_err_ekf': 9999, 'q_err_ekf': 9999,
                 'fc_stats': {'fault_steps': 0, 'fallback_steps': 0,
                              'fault_counts': {}, 'fault_log': []},
                 'solver_stats': {'solve_count': 0, 'fail_count': 0,
                                  'last_info': {'solver': 'none', 'status': 'crash'}}}
        dt = time.time() - t0
        results.append((seed, success, m))
        mark = "OK" if success else "FAIL"
        print("[%3d/%3d] seed=%-5d %s  h=%6.2f  vz=%5.2f  tilt=%4.1f  pos=%5.2f  t=%5.1f  fuel=%5.0f  %4.1fs" %
              (idx + 1, len(seeds), seed, mark,
               m['h_final'], m['vz_final'], m['tilt_final'],
               m['pos_err_final'], m['t_final'], m['fuel_final'], dt))
        sys.stdout.flush()
    return results


def analyze_results(results):
    """分析并打印统计."""
    n_total = len(results)
    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]
    n_success = len(successes)
    n_fail = len(failures)
    success_rate = 100.0 * n_success / n_total

    print("\n" + "=" * 70)
    print("最终100次蒙特卡洛统计 (理论方案2.0 收官)")
    print("=" * 70)

    print("\n--- 成功率 ---")
    print("  成功: %d/%d (%.1f%%)" % (n_success, n_total, success_rate))
    print("  失败: %d" % n_fail)

    if successes:
        h_arr = np.array([m['h_final'] for _, _, m in successes])
        vz_arr = np.array([m['vz_final'] for _, _, m in successes])
        tilt_arr = np.array([m['tilt_final'] for _, _, m in successes])
        pos_arr = np.array([m['pos_err_final'] for _, _, m in successes])
        t_arr = np.array([m['t_final'] for _, _, m in successes])
        fuel_arr = np.array([m['fuel_final'] for _, _, m in successes])
        pe_arr = np.array([m['pos_err_ekf'] for _, _, m in successes])
        ve_arr = np.array([m['vel_err_ekf'] for _, _, m in successes])
        qe_arr = np.array([m['q_err_ekf'] for _, _, m in successes])

        print("\n--- 着陆精度 (成功案例统计) ---")
        print("  指标        均值      标准差     最小      最大      P95")
        print("  h(m)     %7.3f  %7.3f  %7.3f  %7.3f  %7.3f" %
              (h_arr.mean(), h_arr.std(), h_arr.min(), h_arr.max(), np.percentile(h_arr, 95)))
        print("  vz(m/s)  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f" %
              (vz_arr.mean(), vz_arr.std(), vz_arr.min(), vz_arr.max(), np.percentile(vz_arr, 95)))
        print("  tilt(°)  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f" %
              (tilt_arr.mean(), tilt_arr.std(), tilt_arr.min(), tilt_arr.max(), np.percentile(tilt_arr, 95)))
        print("  pos(m)   %7.3f  %7.3f  %7.3f  %7.3f  %7.3f" %
              (pos_arr.mean(), pos_arr.std(), pos_arr.min(), pos_arr.max(), np.percentile(pos_arr, 95)))
        print("  t(s)     %7.3f  %7.3f  %7.3f  %7.3f  %7.3f" %
              (t_arr.mean(), t_arr.std(), t_arr.min(), t_arr.max(), np.percentile(t_arr, 95)))
        print("  fuel(kg) %7.3f  %7.3f  %7.3f  %7.3f  %7.3f" %
              (fuel_arr.mean(), fuel_arr.std(), fuel_arr.min(), fuel_arr.max(), np.percentile(fuel_arr, 95)))

        print("\n--- EKF估计精度 (成功案例) ---")
        print("  位置误差: 均值=%.3fm  P95=%.3fm  最大=%.3fm" %
              (pe_arr.mean(), np.percentile(pe_arr, 95), pe_arr.max()))
        print("  速度误差: 均值=%.3fm/s  P95=%.3fm/s  最大=%.3fm/s" %
              (ve_arr.mean(), np.percentile(ve_arr, 95), ve_arr.max()))
        print("  姿态误差: 均值=%.3f°  P95=%.3f°  最大=%.3f°" %
              (qe_arr.mean(), np.percentile(qe_arr, 95), qe_arr.max()))

    # 求解器统计
    solver_counts = {'CLARABEL': 0, 'SCS': 0, 'none': 0, 'other': 0}
    total_solves = 0
    total_fails = 0
    for _, _, m in results:
        info = m.get('solver_stats', {}).get('last_info', {})
        solver = info.get('solver', 'none')
        if solver in solver_counts:
            solver_counts[solver] += 1
        else:
            solver_counts['other'] += 1
        total_solves += m.get('solver_stats', {}).get('solve_count', 0)
        total_fails += m.get('solver_stats', {}).get('fail_count', 0)

    print("\n--- 求解器统计 ---")
    print("  总求解次数: %d  总失败次数: %d  失败率: %.2f%%" %
          (total_solves, total_fails, 100.0 * total_fails / max(1, total_solves)))
    print("  最终求解器分布: %s" % solver_counts)

    # 故障统计
    total_fault_steps = 0
    total_fallback_steps = 0
    fault_type_counts = {}
    for _, _, m in results:
        fs = m.get('fc_stats', {})
        total_fault_steps += fs.get('fault_steps', 0)
        total_fallback_steps += fs.get('fallback_steps', 0)
        for k, v in fs.get('fault_counts', {}).items():
            fault_type_counts[k] = fault_type_counts.get(k, 0) + v

    print("\n--- 故障/兜底统计 ---")
    print("  总故障步数: %d  总兜底步数: %d" % (total_fault_steps, total_fallback_steps))
    print("  故障分类: %s" % fault_type_counts)

    # 失败案例
    if failures:
        print("\n--- 失败案例分析 ---")
        for seed, _, m in failures[:10]:
            print("  seed=%d: h=%.2fm  vz=%.2fm/s  tilt=%.1f°  pos=%.2fm  t=%.1fs  fuel=%.0fkg" %
                  (seed, m['h_final'], m['vz_final'], m['tilt_final'],
                   m['pos_err_final'], m['t_final'], m['fuel_final']))

    # 最终结论
    print("\n" + "=" * 70)
    print("最终结论")
    print("=" * 70)
    if success_rate >= 95:
        verdict = "优秀 (>=95%)"
    elif success_rate >= 90:
        verdict = "达标 (>=90%)"
    elif success_rate >= 85:
        verdict = "临界 (85-90%)"
    else:
        verdict = "不达标 (<85%)"
    print("  成功率: %.1f%%  →  %s" % (success_rate, verdict))

    if successes:
        print("  P95着陆精度: h=%.2fm  vz=%.2fm/s  tilt=%.2f°  pos=%.2fm" %
              (np.percentile(h_arr, 95), np.percentile(vz_arr, 95),
               np.percentile(tilt_arr, 95), np.percentile(pos_arr, 95)))
        print("\n  对照Falcon 9实际指标:")
        print("    着陆速度: P95=%.2fm/s (F9实际: ~2-3m/s)" % np.percentile(vz_arr, 95))
        print("    着陆姿态: P95=%.2f° (F9实际: ~2-5°)" % np.percentile(tilt_arr, 95))
        print("    着陆点偏差: P95=%.2fm (F9实际: ~1-10m)" % np.percentile(pos_arr, 95))

    # 保存详细结果到txt
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write("100次蒙特卡洛结果摘要\n")
        f.write("=" * 60 + "\n")
        f.write("成功率: %d/%d (%.1f%%)\n\n" % (n_success, n_total, success_rate))
        f.write("seed    success  h(m)    vz(m/s)  tilt(°)  pos(m)  t(s)   fuel(kg)\n")
        f.write("-" * 70 + "\n")
        for seed, success, m in results:
            mark = "OK" if success else "FAIL"
            f.write("%-7d %-7s  %6.2f  %6.2f  %6.2f  %6.2f  %6.1f  %6.0f\n" %
                    (seed, mark, m['h_final'], m['vz_final'], m['tilt_final'],
                     m['pos_err_final'], m['t_final'], m['fuel_final']))
    print("\n详细结果已保存到: %s" % RESULT_FILE)


if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 70)
    print("理论方案2.0 收官: 100次蒙特卡洛验证 (增量保存版)")
    print("=" * 70)

    # 生成100个种子
    all_seeds = [42 + i * 7 for i in range(100)]

    # 加载已有进度
    results, next_idx, saved_seeds, n_total = load_progress()
    if saved_seeds is not None and list(saved_seeds) == list(all_seeds):
        print("断点续跑: 从第 %d 个种子开始" % (next_idx + 1))
    else:
        results = []
        next_idx = 0

    t_start = time.time()

    # 分批运行, 每批10个
    while next_idx < len(all_seeds):
        batch_end = min(next_idx + 10, len(all_seeds))
        print("\n--- 批次 %d-%d ---" % (next_idx + 1, batch_end))
        batch_results = run_batch(all_seeds, next_idx, batch_size=10)
        results.extend(batch_results)
        next_idx = batch_end
        save_progress(results, next_idx, all_seeds, len(all_seeds))
        print("进度已保存: %d/%d" % (len(results), len(all_seeds)))
        sys.stdout.flush()

    t_total = time.time() - t_start
    print("\n总耗时: %.1fs (本会话)" % t_total)

    # 删除进度文件
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    # 分析结果
    analyze_results(results)
