"""
train_all_containers_ratio0.py

以监督比例=0 依次训练四类集装器（AKE / pmc_F_ld / pmc_md11f_md / pge_md11f_md），
输出各集装器独立训练的结果和模型。

用法：
    python train_all_containers_ratio0.py        # 依次训练全部四类
    python train_all_containers_ratio0.py AKE    # 仅训练 AKE（供内部子进程调用）

输出目录：
    ./models_irregular_AKE/ratio_000pct/
    ./models_irregular_pmc_F_ld/ratio_000pct/
    ./models_irregular_pmc_md11f_md/ratio_000pct/
    ./models_irregular_pge_md11f_md/ratio_000pct/
"""

import os
import re
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CONTAINER_TYPES = ['AKE', 'pmc_F_ld', 'pmc_md11f_md', 'pge_md11f_md']
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARAM_FILE = os.path.join(_SCRIPT_DIR, 'parameter_irregular.py')


# ======================================================================
#  参数文件临时替换工具
# ======================================================================

def _read_param_file():
    with open(_PARAM_FILE, 'r', encoding='utf-8') as f:
        return f.read()


def _write_param_file(content):
    with open(_PARAM_FILE, 'w', encoding='utf-8') as f:
        f.write(content)


def _patch_container_type(original_content, container_type):
    """返回将 CONTAINER_TYPE 替换为 container_type 后的文件内容。"""
    return re.sub(
        r"^(CONTAINER_TYPE\s*=\s*)'[^']*'",
        rf"\g<1>'{container_type}'",
        original_content,
        flags=re.MULTILINE,
    )


# ======================================================================
#  单集装器训练（在子进程中运行）
# ======================================================================

def _run_training():
    """
    训练当前 parameter_irregular.py 所配置的集装器，监督比例固定为 0。
    本函数在已完成参数修补的子进程中调用。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.optim as optim
    import torch.multiprocessing as mp

    from parameter_irregular import (
        CONTAINER_TYPE, CONTAINER_SIZE, SAVE_DIR,
        EXCEL_FILE, DATA_SCALE, NUM_PROCESSES, MAX_EPISODES,
        LEARNING_RATE,
    )
    from network_irregular import IrregularA3CNet
    from data_loader import (generate_training_episodes_from_excel,
                             ensure_save_directory)
    from worker import SupervisedWorkerAgent
    from model_saver import ModelSaver

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  集装器  : {CONTAINER_TYPE}  尺寸: {CONTAINER_SIZE}")
    print(f"  监督比例: 0%  (纯强化学习)")
    print(f"  保存目录: {SAVE_DIR}")
    print(sep)

    ratio_dir  = os.path.join(SAVE_DIR, 'ratio_000pct')
    os.makedirs(ratio_dir, exist_ok=True)
    model_path = os.path.join(ratio_dir, 'model.pth')
    model_saver = ModelSaver(model_path)

    try:
        episode_assignments, excel_data = generate_training_episodes_from_excel(
            EXCEL_FILE, DATA_SCALE, NUM_PROCESSES
        )
    except Exception as e:
        print(f"  [错误] 读取数据失败: {e}")
        return None

    all_episode_ids = [ep for eps in episode_assignments for ep in eps]
    labeled_ids     = set()   # ratio=0 → 全部无标签，纯 RL
    print(f"  Episodes 总数 : {len(all_episode_ids)}  (有标签 0 个)")

    mp.set_start_method('spawn', force=True)

    t0         = time.time()
    global_net = IrregularA3CNet(CONTAINER_SIZE)
    global_net.share_memory()
    optimizer  = optim.Adam(global_net.parameters(), lr=LEARNING_RATE)

    global_ep_idx = mp.Value('i', 0)
    global_ep_r   = mp.Value('d', 0.)
    res_queue     = mp.Queue()

    workers = [
        SupervisedWorkerAgent(
            global_net, optimizer, global_ep_idx, global_ep_r,
            res_queue, i, episode_assignments[i], excel_data,
            model_saver, labeled_ids,
        )
        for i in range(NUM_PROCESSES)
    ]
    [w.start() for w in workers]

    results  = []
    done_cnt = 0
    try:
        while done_cnt < NUM_PROCESSES:
            item = res_queue.get()
            if item is None:
                done_cnt += 1
            else:
                results.append(item)
    except KeyboardInterrupt:
        print(f"\n  [{CONTAINER_TYPE}] 训练被中断")
        [w.terminate() for w in workers]
    [w.join() for w in workers]

    model_saver.save_model(global_net.state_dict(), MAX_EPISODES)

    if not results:
        print(f"  [{CONTAINER_TYPE}] 无训练结果")
        return None

    elapsed = int(time.time() - t0)
    rewards  = [r[2] for r in results]
    utils    = [r[3] for r in results]
    placeds  = [r[4] for r in results]

    stats = {
        'container_type'     : CONTAINER_TYPE,
        'total_seconds'      : elapsed,
        'total_cases'        : len(results),
        'avg_utilization'    : float(np.mean(utils)),
        'avg_reward_last50'  : float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards)),
        'max_reward'         : float(max(rewards)),
        'avg_items'          : float(np.mean(placeds)),
        'model_path'         : model_path,
        'ratio_dir'          : ratio_dir,
    }

    _save_curve(results, CONTAINER_TYPE, ratio_dir)
    _save_result(stats, ratio_dir, EXCEL_FILE)
    _print_stats(stats)
    return stats


# ======================================================================
#  输出工具
# ======================================================================

def _save_curve(results, container_type, ratio_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    episodes = [r[1] for r in results]
    rewards  = [r[2] for r in results]
    utils    = [r[3] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{container_type}  监督比例 0%', fontsize=13)

    axes[0].plot(episodes, rewards, alpha=0.5, linewidth=0.8)
    if len(rewards) >= 50:
        ma = [np.mean(rewards[max(0, i - 50):i + 1]) for i in range(len(rewards))]
        axes[0].plot(episodes, ma, linewidth=1.5, label='MA-50')
        axes[0].legend()
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Reward')
    axes[0].set_title('奖励曲线'); axes[0].grid(True)

    axes[1].plot(episodes, [u * 100 for u in utils], alpha=0.5, linewidth=0.8, color='orange')
    if len(utils) >= 50:
        ma_u = [np.mean(utils[max(0, i - 50):i + 1]) * 100 for i in range(len(utils))]
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
    print(f"  [{s['container_type']}]  监督比例 0%  训练完成")
    print(sep)
    print(f"  训练时间          : {h}h {m}m {sec}s")
    print(f"  训练案例数        : {s['total_cases']}")
    print(f"  平均装载率        : {s['avg_utilization'] * 100:.2f}%")
    print(f"  平均奖励(后50)    : {s['avg_reward_last50']:.2f}")
    print(f"  最高奖励          : {s['max_reward']:.2f}")
    print(f"  平均装载货物数    : {s['avg_items']:.2f}")
    print(f"  模型路径          : {s['model_path']}")
    print(sep)


def _save_result(stats, ratio_dir, excel_file):
    h, rem = divmod(stats['total_seconds'], 3600)
    m, sec = divmod(rem, 60)
    lines = [
        "=" * 60,
        f"[{stats['container_type']}]  监督比例 0%  训练结果",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"数据集  : {excel_file}",
        "=" * 60,
        f"  训练时间          : {h}h {m}m {sec}s",
        f"  训练案例数        : {stats['total_cases']}",
        f"  平均装载率        : {stats['avg_utilization'] * 100:.2f}%",
        f"  平均奖励(后50)    : {stats['avg_reward_last50']:.2f}",
        f"  最高奖励          : {stats['max_reward']:.2f}",
        f"  平均装载货物数    : {stats['avg_items']:.2f}",
        f"  模型路径          : {stats['model_path']}",
        "=" * 60,
    ]
    result_path = os.path.join(ratio_dir, 'result.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  训练结果 → {result_path}")


def _save_all_summary(all_stats):
    """打印并保存四类集装器的横向对比汇总。"""
    valid = [s for s in all_stats if s is not None]
    if not valid:
        return

    sep     = "=" * 70
    header  = (f"{'集装器':^20} | {'装载率':>8} | {'平均奖励(后50)':>14} "
               f"| {'最高奖励':>10} | {'平均货物数':>10}")
    divider = "-" * 70

    print(f"\n\n{sep}")
    print("  四类集装器  监督比例0%  训练汇总")
    print(sep)
    print(header)
    print(divider)
    for s in valid:
        print(f"  {s['container_type']:<18} | "
              f"{s['avg_utilization'] * 100:>7.2f}% | "
              f"{s['avg_reward_last50']:>14.2f} | "
              f"{s['max_reward']:>10.2f} | "
              f"{s['avg_items']:>10.2f}")
    print(sep)

    best = max(valid, key=lambda s: s['avg_utilization'])
    print(f"\n  最高装载率集装器 → {best['container_type']}  "
          f"（装载率 {best['avg_utilization'] * 100:.2f}%）")

    summary_path = os.path.join(_SCRIPT_DIR, 'all_containers_ratio0_summary.txt')
    lines = [
        "四类集装器  监督比例0%  训练汇总报告",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        header,
        divider,
    ]
    for s in valid:
        lines.append(
            f"  {s['container_type']:<18} | "
            f"{s['avg_utilization'] * 100:>7.2f}% | "
            f"{s['avg_reward_last50']:>14.2f} | "
            f"{s['max_reward']:>10.2f} | "
            f"{s['avg_items']:>10.2f}"
        )
    lines += [
        sep,
        f"最高装载率集装器 → {best['container_type']}  "
        f"（装载率 {best['avg_utilization'] * 100:.2f}%）",
        "",
        "各集装器模型路径：",
    ]
    for s in valid:
        lines.append(f"  {s['container_type']:<20}: {s['model_path']}")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  汇总文件 → {summary_path}")


# ======================================================================
#  入口
# ======================================================================

if __name__ == '__main__':

    # ── 子进程模式：由主进程带容器类型参数调用 ──────────────────────
    if len(sys.argv) == 2 and sys.argv[1] in CONTAINER_TYPES:
        _run_training()
        sys.exit(0)

    # ── 主进程模式：依次为四类集装器启动子进程 ─────────────────────
    import subprocess

    print("=" * 60)
    print("  四类集装器  监督比例=0  依次训练")
    print(f"  集装器列表: {CONTAINER_TYPES}")
    print("=" * 60)

    original_content = _read_param_file()
    all_stats        = []
    t_total          = time.time()

    try:
        for ctype in CONTAINER_TYPES:
            print(f"\n{'─'*60}")
            print(f"  开始训练: {ctype}")
            print(f"{'─'*60}")

            patched = _patch_container_type(original_content, ctype)
            _write_param_file(patched)

            proc = subprocess.run(
                [sys.executable, __file__, ctype],
                cwd=_SCRIPT_DIR,
            )

            if proc.returncode != 0:
                print(f"  [{ctype}] 子进程异常退出 (code={proc.returncode})")
                all_stats.append(None)
                continue

            # 读取该集装器的 result.txt，提取关键数据用于最终汇总
            result_txt = os.path.join(
                _SCRIPT_DIR,
                f'models_irregular_{ctype}',
                'ratio_000pct',
                'result.txt',
            )
            stats = {'container_type': ctype, 'model_path': os.path.join(
                _SCRIPT_DIR, f'models_irregular_{ctype}', 'ratio_000pct', 'model.pth'
            )}
            if os.path.exists(result_txt):
                with open(result_txt, encoding='utf-8') as f:
                    txt = f.read()
                def _extract(label):
                    m = re.search(re.escape(label) + r'\s*:\s*([-\d.]+)', txt)
                    return float(m.group(1)) if m else 0.0
                stats['avg_utilization']   = _extract('平均装载率') / 100.0
                stats['avg_reward_last50'] = _extract('平均奖励(后50)')
                stats['max_reward']        = _extract('最高奖励')
                stats['avg_items']         = _extract('平均装载货物数')
                stats['total_cases']       = int(_extract('训练案例数'))
                stats['total_seconds']     = 0
            all_stats.append(stats)

    finally:
        _write_param_file(original_content)
        print(f"\n  parameter_irregular.py 已恢复原始配置")

    elapsed     = int(time.time() - t_total)
    h, rem      = divmod(elapsed, 3600)
    m_tot, sec  = divmod(rem, 60)
    print(f"\n  全部训练完成，总耗时 {h}h {m_tot}m {sec}s")

    _save_all_summary(all_stats)
