"""
批量测试所有监督比例模型
- 数据集: 数据_101010_our20%.xlsx
- 模型: modelscut2/ratio_XXXpct/model.pth (0%~100%)
- 输出: 测试总时间、平均每个案例测试时间、箱子装载率、平均每个箱子装载货物个数
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np

from parameter_irregular import CONTAINER_SIZE, SAVE_DIR, DATA_SCALE, LOOKAHEAD_K
from network_irregular import IrregularA3CNet as ImprovedA3CNet
from data_loader import load_excel_data
from test_model import ModelTesterWithLookahead

TEST_EXCEL = '数据_101010_our20%.xlsx'
RATIOS = [round(i / 10, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0


def test_one_ratio(ratio, data, unique_episodes):
    """用指定比例的模型测试所有episode，返回统计指标"""
    ratio_pct = int(round(ratio * 100))
    model_path = os.path.join(SAVE_DIR, f'ratio_{ratio_pct:03d}pct', 'model.pth')

    if not os.path.exists(model_path):
        print(f"  [跳过] 模型文件不存在: {model_path}")
        return None

    print(f"\n{'='*60}")
    print(f"测试模型  [监督比例 = {ratio_pct}%]  {model_path}")
    print(f"{'='*60}")

    try:
        tester = ModelTesterWithLookahead(model_path=model_path, k=LOOKAHEAD_K)
    except Exception as e:
        print(f"  [错误] 加载模型失败: {e}")
        return None

    t_start = time.time()

    total_placed = 0
    total_items = 0
    total_utilization = 0.0
    util_count = 0

    for ep_id in unique_episodes:
        ep_mask = data[:, 0] == ep_id
        ep_data = data[ep_mask]
        if len(ep_data) == 0:
            continue

        test_items = []
        for row in ep_data:
            data_id = int(row[0])
            l, w, h = int(row[1]), int(row[2]), int(row[3])
            test_items.append((data_id, l, w, h))

        try:
            results = tester.test_multiple_items_with_lookahead(test_items)
        except Exception as e:
            print(f"  [警告] Episode {ep_id} 测试出错: {e}")
            continue

        for r in results:
            total_items += 1
            if r.get('success', False):
                total_placed += 1

        # 取最后一个有利用率的结果
        for r in reversed(results):
            if 'container_utilization' in r:
                total_utilization += r['container_utilization']
                util_count += 1
                break

    t_end = time.time()
    total_sec = t_end - t_start
    n_episodes = util_count  # 成功完成的episode数

    avg_sec_per_case = total_sec / n_episodes if n_episodes > 0 else 0
    avg_utilization = total_utilization / util_count if util_count > 0 else 0
    avg_items_per_box = total_placed / n_episodes if n_episodes > 0 else 0

    stats = {
        'ratio_pct': ratio_pct,
        'total_sec': total_sec,
        'n_episodes': n_episodes,
        'avg_sec_per_case': avg_sec_per_case,
        'avg_utilization': avg_utilization,
        'avg_items_per_box': avg_items_per_box,
        'total_placed': total_placed,
        'total_items': total_items,
    }
    return stats


def print_stats(s):
    h = int(s['total_sec']) // 3600
    m = (int(s['total_sec']) % 3600) // 60
    sec = int(s['total_sec']) % 60
    print(f"\n{'='*60}")
    print(f"测试结果  [监督比例 = {s['ratio_pct']}%]")
    print(f"{'='*60}")
    print(f"  测试总时间            : {h}h {m}m {sec}s  ({s['total_sec']:.2f}s)")
    print(f"  平均每个案例测试时间  : {s['avg_sec_per_case']:.3f} 秒/案例")
    print(f"  箱子装载率(利用率)    : {s['avg_utilization'] * 100:.2f}%")
    print(f"  平均每个箱子装载货物数: {s['avg_items_per_box']:.2f} 个")
    print(f"  测试案例数            : {s['n_episodes']} 个")
    print(f"  成功放置/总物品       : {s['total_placed']}/{s['total_items']}")
    print(f"{'='*60}")


def save_single_result(s, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f'result_ratio_{s["ratio_pct"]:03d}pct.txt')
    h = int(s['total_sec']) // 3600
    m = (int(s['total_sec']) % 3600) // 60
    sec = int(s['total_sec']) % 60
    lines = [
        f"测试结果  [监督比例 = {s['ratio_pct']}%]",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"  测试总时间            : {h}h {m}m {sec}s  ({s['total_sec']:.2f}s)",
        f"  平均每个案例测试时间  : {s['avg_sec_per_case']:.3f} 秒/案例",
        f"  箱子装载率(利用率)    : {s['avg_utilization'] * 100:.2f}%",
        f"  平均每个箱子装载货物数: {s['avg_items_per_box']:.2f} 个",
        f"  测试案例数            : {s['n_episodes']} 个",
        f"  成功放置/总物品       : {s['total_placed']}/{s['total_items']}",
    ]
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  [已保存] {result_path}")


def save_summary(all_stats, save_dir, excel_file):
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, 'test_all_ratios_summary.txt')
    lines = []
    lines.append("全监督比例模型测试汇总报告")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"测试数据集: {excel_file}")
    lines.append("")
    for s in all_stats:
        if s is None:
            continue
        h = int(s['total_sec']) // 3600
        m = (int(s['total_sec']) % 3600) // 60
        sec = int(s['total_sec']) % 60
        lines.append("=" * 60)
        lines.append(f"测试结果  [监督比例 = {s['ratio_pct']}%]")
        lines.append("=" * 60)
        lines.append(f"  测试总时间            : {h}h {m}m {sec}s  ({s['total_sec']:.2f}s)")
        lines.append(f"  平均每个案例测试时间  : {s['avg_sec_per_case']:.3f} 秒/案例")
        lines.append(f"  箱子装载率(利用率)    : {s['avg_utilization'] * 100:.2f}%")
        lines.append(f"  平均每个箱子装载货物数: {s['avg_items_per_box']:.2f} 个")
        lines.append(f"  测试案例数            : {s['n_episodes']} 个")
        lines.append(f"  成功放置/总物品       : {s['total_placed']}/{s['total_items']}")
        lines.append("")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n汇总结果已保存到: {summary_path}")
    return summary_path


if __name__ == '__main__':
    print("=" * 60)
    print(f"全监督比例模型测试  数据集: {TEST_EXCEL}")
    print("=" * 60)

    # 加载测试数据（只加载一次）
    try:
        data = load_excel_data(TEST_EXCEL, DATA_SCALE)
    except Exception as e:
        print(f"读取数据集失败: {e}")
        exit(1)

    unique_episodes = np.unique(data[:, 0])
    print(f"共 {len(unique_episodes)} 个 episode")

    all_stats = []
    for ratio in RATIOS:
        stats = test_one_ratio(ratio, data, unique_episodes)
        all_stats.append(stats)
        if stats is not None:
            print_stats(stats)
            save_single_result(stats, SAVE_DIR)

    # 汇总打印
    print("\n\n" + "=" * 60)
    print("所有比例测试完成！汇总结果：")
    print("=" * 60)
    valid_stats = [s for s in all_stats if s is not None]

    # 保存汇总文件
    save_summary(valid_stats, SAVE_DIR, TEST_EXCEL)
