"""
训练主函数：依次以 0%, 10%, 20%, ..., 100% 监督比例训练 11 组实验。

监督比例含义：
  该比例的 episode 使用专家位置标签（NLL Loss 向专家位置靠拢）
  其余 episode 使用纯 RL（标准 A3C Loss）
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import time
import numpy as np

from parameter_irregular import *
from network import ImprovedA3CNet
from data_loader import (generate_training_episodes_from_excel,
                         ensure_save_directory, get_labeled_episode_ids)
from worker import SupervisedWorkerAgent
from model_saver import ModelSaver


# ======================================================================
#  单组实验
# ======================================================================

def train_one_ratio(episode_assignments, excel_data, all_episode_ids, supervised_ratio):
    """
    以指定监督比例训练一次，返回 stats dict。
    supervised_ratio : 0.0 ~ 1.0
    """
    ratio_pct = int(round(supervised_ratio * 100))
    sep = "=" * 60
    print(f"\n{sep}\n  开始训练  [监督比例 = {ratio_pct}%]\n{sep}")

    ratio_dir  = os.path.join(SAVE_DIR, f'ratio_{ratio_pct:03d}pct')
    os.makedirs(ratio_dir, exist_ok=True)
    model_path = os.path.join(ratio_dir, 'model.pth')
    model_saver = ModelSaver(model_path)

    labeled_ids = get_labeled_episode_ids(all_episode_ids, supervised_ratio)
    print(f"  有标签 episodes : {len(labeled_ids)} / {len(all_episode_ids)}")

    t0 = time.time()

    global_net = ImprovedA3CNet(CONTAINER_SIZE)
    global_net.share_memory()
    optimizer  = optim.Adam(global_net.parameters(), lr=LEARNING_RATE)

    global_ep_idx = mp.Value('i', 0)
    global_ep_r   = mp.Value('d', 0.)
    res_queue     = mp.Queue()

    workers = [
        SupervisedWorkerAgent(global_net, optimizer, global_ep_idx, global_ep_r,
                              res_queue, i, episode_assignments[i], excel_data,
                              model_saver, labeled_ids)
        for i in range(NUM_PROCESSES)
    ]

    [w.start() for w in workers]

    results = []
    done_cnt = 0
    try:
        while done_cnt < NUM_PROCESSES:
            item = res_queue.get()
            if item is None:
                done_cnt += 1
            else:
                results.append(item)
    except KeyboardInterrupt:
        print(f"\n[{ratio_pct}%] 训练被中断")
        [w.terminate() for w in workers]
    [w.join() for w in workers]

    model_saver.save_model(global_net.state_dict(), MAX_EPISODES)

    if not results:
        print(f"[{ratio_pct}%] 无训练结果")
        return None

    elapsed   = int(time.time() - t0)
    rewards   = [r[2] for r in results]
    utils     = [r[3] for r in results]
    placeds   = [r[4] for r in results]

    stats = {
        'ratio_pct'          : ratio_pct,
        'total_seconds'      : elapsed,
        'total_cases'        : len(results),
        'avg_utilization'    : float(np.mean(utils)),
        'avg_reward_last50'  : float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards)),
        'max_reward'         : float(max(rewards)),
        'avg_items'          : float(np.mean(placeds)),
        'model_path'         : model_path,
    }

    _save_curve(results, ratio_pct, ratio_dir)
    _save_ratio_result(stats, ratio_dir)
    _print_stats(stats)
    return stats


# ======================================================================
#  输出工具
# ======================================================================

def _save_curve(results, ratio_pct, ratio_dir):
    episodes = [r[1] for r in results]
    rewards  = [r[2] for r in results]
    utils    = [r[3] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'监督比例 {ratio_pct}%', fontsize=13)

    axes[0].plot(episodes, rewards, alpha=0.5, linewidth=0.8)
    if len(rewards) >= 50:
        ma = [np.mean(rewards[max(0, i-50):i+1]) for i in range(len(rewards))]
        axes[0].plot(episodes, ma, linewidth=1.5, label='MA-50')
        axes[0].legend()
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Reward')
    axes[0].set_title('奖励曲线'); axes[0].grid(True)

    axes[1].plot(episodes, [u * 100 for u in utils], alpha=0.5, linewidth=0.8, color='orange')
    if len(utils) >= 50:
        ma_u = [np.mean(utils[max(0, i-50):i+1]) * 100 for i in range(len(utils))]
        axes[1].plot(episodes, ma_u, linewidth=1.5, color='darkorange', label='MA-50')
        axes[1].legend()
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('利用率 (%)')
    axes[1].set_title('装载率曲线'); axes[1].grid(True)

    wdata = {}
    for r in results:
        wn = r[0]
        wdata.setdefault(wn, {'ep': [], 'r': []})
        wdata[wn]['ep'].append(r[1]); wdata[wn]['r'].append(r[2])
    for wn, d in wdata.items():
        axes[2].plot(d['ep'], d['r'], label=wn, alpha=0.7, linewidth=0.8)
    axes[2].set_xlabel('Episode'); axes[2].set_ylabel('Reward')
    axes[2].set_title('各 Worker 奖励'); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    path = os.path.join(ratio_dir, 'training_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  训练曲线 → {path}")


def _print_stats(s):
    h, rem = divmod(s['total_seconds'], 3600)
    m, sec = divmod(rem, 60)
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  结果汇总  [监督比例 {s['ratio_pct']}%]")
    print(f"{sep}")
    print(f"  训练时间          : {h}h {m}m {sec}s")
    print(f"  训练案例数        : {s['total_cases']}")
    print(f"  平均装载率        : {s['avg_utilization']*100:.2f}%")
    print(f"  平均奖励(后50)    : {s['avg_reward_last50']:.2f}")
    print(f"  最高奖励          : {s['max_reward']:.2f}")
    print(f"  平均装载货物数    : {s['avg_items']:.2f}")
    print(f"{sep}")


def _save_ratio_result(stats, ratio_dir):
    h, rem = divmod(stats['total_seconds'], 3600)
    m, sec = divmod(rem, 60)
    lines = [
        "=" * 60,
        f"监督比例 {stats['ratio_pct']}%  训练结果",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"数据集  : {EXCEL_FILE}",
        "=" * 60,
        f"  训练时间          : {h}h {m}m {sec}s",
        f"  训练案例数        : {stats['total_cases']}",
        f"  平均装载率        : {stats['avg_utilization']*100:.2f}%",
        f"  平均奖励(后50)    : {stats['avg_reward_last50']:.2f}",
        f"  最高奖励          : {stats['max_reward']:.2f}",
        f"  平均装载货物数    : {stats['avg_items']:.2f}",
        f"  模型路径          : {stats['model_path']}",
        "=" * 60,
    ]
    with open(os.path.join(ratio_dir, 'result.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _save_summary(all_stats):
    """输出所有比例的横向对比表格，并保存汇总文件。"""
    valid = [s for s in all_stats if s is not None]
    if not valid:
        return

    sep = "=" * 75
    header = f"{'比例':>6} | {'装载率':>8} | {'平均奖励(后50)':>14} | {'最高奖励':>10} | {'平均货物数':>10}"
    divider = "-" * 75

    print(f"\n\n{sep}")
    print("  全比例对比汇总")
    print(sep)
    print(header)
    print(divider)
    for s in valid:
        print(f"  {s['ratio_pct']:>3}%  | "
              f"{s['avg_utilization']*100:>7.2f}% | "
              f"{s['avg_reward_last50']:>14.2f} | "
              f"{s['max_reward']:>10.2f} | "
              f"{s['avg_items']:>10.2f}")
    print(sep)

    # 找最优比例（以平均装载率为准）
    best = max(valid, key=lambda s: s['avg_utilization'])
    print(f"\n  最优监督比例 → {best['ratio_pct']}%  "
          f"（装载率 {best['avg_utilization']*100:.2f}%）")

    # 保存汇总文件
    summary_path = os.path.join(SAVE_DIR, 'all_ratios_summary.txt')
    lines = [
        "全比例训练汇总报告",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"数据集  : {EXCEL_FILE}",
        "",
        header,
        divider,
    ]
    for s in valid:
        lines.append(
            f"  {s['ratio_pct']:>3}%  | "
            f"{s['avg_utilization']*100:>7.2f}% | "
            f"{s['avg_reward_last50']:>14.2f} | "
            f"{s['max_reward']:>10.2f} | "
            f"{s['avg_items']:>10.2f}"
        )
    lines += [
        sep,
        f"最优监督比例 → {best['ratio_pct']}%  （装载率 {best['avg_utilization']*100:.2f}%）",
    ]
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  汇总文件 → {summary_path}")

    # 绘制对比折线图
    ratios = [s['ratio_pct'] for s in valid]
    utils  = [s['avg_utilization'] * 100 for s in valid]
    rews   = [s['avg_reward_last50'] for s in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('各监督比例训练结果对比', fontsize=13)

    ax1.plot(ratios, utils, 'o-', color='steelblue')
    ax1.set_xlabel('监督比例 (%)'); ax1.set_ylabel('平均装载率 (%)')
    ax1.set_title('装载率 vs 监督比例'); ax1.grid(True)
    for r, u in zip(ratios, utils):
        ax1.annotate(f'{u:.1f}%', (r, u), textcoords='offset points', xytext=(0, 6), fontsize=8)

    ax2.plot(ratios, rews, 'o-', color='darkorange')
    ax2.set_xlabel('监督比例 (%)'); ax2.set_ylabel('平均奖励 (后50 eps)')
    ax2.set_title('平均奖励 vs 监督比例'); ax2.grid(True)
    for r, rw in zip(ratios, rews):
        ax2.annotate(f'{rw:.1f}', (r, rw), textcoords='offset points', xytext=(0, 6), fontsize=8)

    plt.tight_layout()
    cmp_path = os.path.join(SAVE_DIR, 'ratio_comparison.png')
    plt.savefig(cmp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图表 → {cmp_path}")


# ======================================================================
#  入口
# ======================================================================

if __name__ == '__main__':
    print("A3C 三维装箱训练  —  0%~100% 监督比例对比实验")
    print("=" * 60)

    from test import test_excel_data_loading_corrected
    if not test_excel_data_loading_corrected():
        print("数据加载失败，请检查 Excel 文件和配置")
        exit(1)

    if not ensure_save_directory(SAVE_DIR):
        print("无法创建保存目录，训练终止")
        exit(1)

    try:
        episode_assignments, excel_data = generate_training_episodes_from_excel(
            EXCEL_FILE, DATA_SCALE, NUM_PROCESSES
        )
        all_episode_ids = [ep for eps in episode_assignments for ep in eps]
        print(f"\n数据加载完成：共 {len(all_episode_ids)} 个 episodes")
    except Exception as e:
        print(f"读取 Excel 数据失败: {e}")
        exit(1)

    mp.set_start_method('spawn', force=True)

    ratios    = [0.3]   # 0.3
    all_stats = []
    t_total   = time.time()

    print(f"\n即将依次训练 {len(ratios)} 组实验：")
    print("  " + "  ".join(f"{int(r*100)}%" for r in ratios))

    for ratio in ratios:
        stats = train_one_ratio(episode_assignments, excel_data, all_episode_ids, ratio)
        all_stats.append(stats)

    elapsed = int(time.time() - t_total)
    h, rem  = divmod(elapsed, 3600)
    m, sec  = divmod(rem, 60)
    print(f"\n所有实验完成，总耗时 {h}h {m}m {sec}s")

    _save_summary(all_stats)
