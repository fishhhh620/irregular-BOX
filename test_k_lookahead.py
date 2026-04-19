"""
前瞻窗口 K=1~8 批量测试脚本
- 模型 : modelscut2/ratio_030pct/model.pth  (10% 监督比例)
- K 值 : 1, 2, 3, 4, 5, 6, 7, 8
- 数据集: 数据_101010_cut1.xlsx
          数据_101010_cut2.xlsx
          数据_101010_rs.xlsx
- 输出 : 每完成一个 K 值的全部 3 个数据集立即打印 + 保存结果
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np

from parameter_irregular import CONTAINER_SIZE, SAVE_DIR, DATA_SCALE
from data_loader import load_excel_data
from test_model import ModelTesterWithLookahead

# ── 配置 ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(SAVE_DIR, 'ratio_030pct', 'model.pth')
K_VALUES     = list(range(1, 9))           # 1 ~ 8
TEST_FILES   = [
    '数据_101010_cut1.xlsx',
    '数据_101010_cut2.xlsx',
    '数据_101010_rs.xlsx',
]
RESULT_DIR   = os.path.join(SAVE_DIR, 'k_lookahead_test')
# ─────────────────────────────────────────────────────────────────────────────


def load_datasets():
    """预加载所有测试数据集，避免重复 IO。返回 {filename: (data, unique_episodes)}"""
    datasets = {}
    for fname in TEST_FILES:
        print(f"  加载数据: {fname} ...", end=' ', flush=True)
        try:
            data = load_excel_data(fname, DATA_SCALE)
            unique_eps = np.unique(data[:, 0])
            datasets[fname] = (data, unique_eps)
            print(f"OK  ({len(unique_eps)} episodes)")
        except Exception as e:
            print(f"失败: {e}")
            datasets[fname] = None
    return datasets


def test_one_dataset(tester, fname, data, unique_episodes):
    """用已加载的 tester 测试单个数据集，返回统计字典。"""
    t_start = time.time()

    total_placed    = 0
    total_items     = 0
    total_util      = 0.0
    util_count      = 0

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
            print(f"    [警告] Episode {ep_id} 出错: {e}")
            continue

        for r in results:
            total_items += 1
            if r.get('success', False):
                total_placed += 1

        # 取本 episode 最后一个含利用率的结果
        for r in reversed(results):
            if 'container_utilization' in r:
                total_util  += r['container_utilization']
                util_count  += 1
                break

    total_sec = time.time() - t_start
    n_ep      = util_count

    return {
        'fname'              : fname,
        'total_sec'          : total_sec,
        'n_episodes'         : n_ep,
        'avg_sec_per_case'   : total_sec / n_ep if n_ep > 0 else 0.0,
        'avg_utilization'    : total_util / util_count if util_count > 0 else 0.0,
        'avg_items_per_box'  : total_placed / n_ep if n_ep > 0 else 0.0,
        'total_placed'       : total_placed,
        'total_items'        : total_items,
    }


def fmt_time(sec):
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    return f"{h}h {m}m {s}s ({sec:.2f}s)"


def print_k_results(k, stats_list):
    """打印一个 K 值下所有数据集的结果。"""
    print(f"\n{'='*65}")
    print(f"K = {k}")
    print(f"{'='*65}")
    for s in stats_list:
        print(f"\n{s['fname']}")
        print(f"  测试总时间            : {fmt_time(s['total_sec'])}")
        print(f"  平均每个案例测试时间  : {s['avg_sec_per_case']:.3f} 秒/案例")
        print(f"  箱子装载率(利用率)    : {s['avg_utilization']*100:.2f}%")
        print(f"  平均每个箱子装载货物数: {s['avg_items_per_box']:.2f} 个")
        print(f"  测试案例数            : {s['n_episodes']}")
        print(f"  成功放置/总物品       : {s['total_placed']}/{s['total_items']}")


def save_k_results(k, stats_list, result_dir):
    """将一个 K 值的结果追加写入文件。"""
    os.makedirs(result_dir, exist_ok=True)
    fpath = os.path.join(result_dir, f'result_k{k:02d}.txt')
    lines = [
        f"K = {k}",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
    ]
    for s in stats_list:
        lines += [
            f"\n{s['fname']}",
            f"  测试总时间            : {fmt_time(s['total_sec'])}",
            f"  平均每个案例测试时间  : {s['avg_sec_per_case']:.3f} 秒/案例",
            f"  箱子装载率(利用率)    : {s['avg_utilization']*100:.2f}%",
            f"  平均每个箱子装载货物数: {s['avg_items_per_box']:.2f} 个",
            f"  测试案例数            : {s['n_episodes']}",
            f"  成功放置/总物品       : {s['total_placed']}/{s['total_items']}",
        ]
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  [已保存] {fpath}")


def save_summary(all_k_stats, result_dir):
    """所有 K 值测试完后保存汇总文件。"""
    os.makedirs(result_dir, exist_ok=True)
    fpath = os.path.join(result_dir, 'summary_all_k.txt')
    lines = [
        "前瞻窗口 K=1~8 测试汇总",
        f"模型: {MODEL_PATH}",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for k, stats_list in sorted(all_k_stats.items()):
        lines.append("=" * 65)
        lines.append(f"K = {k}")
        lines.append("=" * 65)
        for s in stats_list:
            lines += [
                f"\n{s['fname']}",
                f"  测试总时间            : {fmt_time(s['total_sec'])}",
                f"  平均每个案例测试时间  : {s['avg_sec_per_case']:.3f} 秒/案例",
                f"  箱子装载率(利用率)    : {s['avg_utilization']*100:.2f}%",
                f"  平均每个箱子装载货物数: {s['avg_items_per_box']:.2f} 个",
                f"  测试案例数            : {s['n_episodes']}",
                f"  成功放置/总物品       : {s['total_placed']}/{s['total_items']}",
            ]
        lines.append("")
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n汇总结果已保存: {fpath}")


if __name__ == '__main__':
    print("=" * 65)
    print(f"前瞻窗口 K=1~8 测试  (10% 监督比例模型)")
    print(f"模型路径: {MODEL_PATH}")
    print("=" * 65)

    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        exit(1)

    # 预加载所有数据集
    print("\n预加载数据集...")
    datasets = load_datasets()

    all_k_stats = {}   # {k: [stats_cut1, stats_cut2, stats_rs]}

    for k in K_VALUES:
        print(f"\n{'─'*65}")
        print(f"开始测试  K = {k}")
        print(f"{'─'*65}")

        # 为当前 K 值创建 tester（每次 K 变化重新创建）
        try:
            tester = ModelTesterWithLookahead(model_path=MODEL_PATH, k=k)
        except Exception as e:
            print(f"[错误] 加载模型失败 (K={k}): {e}")
            continue

        k_stats = []
        for fname in TEST_FILES:
            if datasets[fname] is None:
                print(f"  [跳过] {fname} (数据加载失败)")
                continue

            data, unique_eps = datasets[fname]
            print(f"\n  测试: {fname}  (K={k}, {len(unique_eps)} episodes)")
            stats = test_one_dataset(tester, fname, data, unique_eps)
            k_stats.append(stats)

        all_k_stats[k] = k_stats

        # 每完成一个 K 值立即输出并保存
        print_k_results(k, k_stats)
        save_k_results(k, k_stats, RESULT_DIR)

    # 全部完成后保存汇总
    save_summary(all_k_stats, RESULT_DIR)
    print("\n全部测试完成。")
