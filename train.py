"""
训练主函数：依次以 0%, 10%, 20%, ..., 100% 监督比例训练 11 组实验。

监督比例含义：
  该比例的 episode 使用专家位置标签（NLL Loss 向专家位置靠拢）
  其余 episode 使用纯 RL（标准 A3C Loss）

用法：
  python train.py                  # 依次训练全部四类集装器，每类跑 0%~100% 监督比例
  python train.py pmc_md11f_md     # 仅训练指定集装器（子进程模式，主进程内部调用）
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import time
import numpy as np

from parameter_irregular import *
from network_irregular import IrregularA3CNet as ImprovedA3CNet
from data_loader import (generate_training_episodes_from_excel,
                         ensure_save_directory, get_labeled_episode_ids)
from worker import SupervisedWorkerAgent
from model_saver import ModelSaver

# 四类集装器（与 parameter_irregular.py 中定义的一致）
CONTAINER_TYPES = ['AKE', 'pmc_F_ld', 'pmc_md11f_md', 'pge_md11f_md']
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARAM_FILE = os.path.join(_SCRIPT_DIR, 'parameter_irregular.py')


# ======================================================================
#  参数文件临时替换工具（与 train_all_containers_ratio0.py 相同）
# ======================================================================

def _read_param_file():
    with open(_PARAM_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def _write_param_file(content):
    with open(_PARAM_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

def _patch_container_type(original_content, container_type):
    return re.sub(
        r"^(CONTAINER_TYPE\s*=\s*)'[^']*'",
        rf"\g<1>'{container_type}'",
        original_content,
        flags=re.MULTILINE,
    )


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

    # 装载图：用训练好的模型跑一个贪心评估 episode 并可视化
    try:
        eval_container = _run_eval_episode(model_path, excel_data, all_episode_ids[0])
        _save_loading_diagram(eval_container, ratio_pct, ratio_dir)
    except Exception as e:
        print(f"  [装载图] 生成失败: {e}")

    return stats


# ======================================================================
#  装载图：评估 episode + 3D 可视化
# ======================================================================

def _run_eval_episode(model_path, excel_data, episode_id):
    """用训练好的模型贪心地跑一个 episode，返回装载后的容器对象。"""
    from data_loader import extract_episode_data_corrected, OnlineItemIterator
    from environment_irregular import IrregularBinPackingEnv

    try:
        net = ImprovedA3CNet(CONTAINER_SIZE)
        state_dict = torch.load(model_path, map_location='cpu')
        net.load_state_dict(state_dict)
        net.eval()
    except Exception as e:
        print(f"  [装载图] 模型加载失败: {e}")
        return None

    try:
        items_data, _, _ = extract_episode_data_corrected(excel_data, episode_id)
    except Exception as e:
        print(f"  [装载图] 数据提取失败: {e}")
        return None

    if not items_data:
        return None

    item_iterator = OnlineItemIterator(items_data, [], [])
    env = IrregularBinPackingEnv(CONTAINER_TYPE, [])
    env.reset()

    L, W, H = CONTAINER_SIZE

    while item_iterator.has_next():
        result = item_iterator.get_next_item()
        if result is None:
            break
        item_data, _, _ = result

        k = max(1, min(LOOKAHEAD_K, K_MAX))
        lah = item_iterator.peek_next_items(max(0, k - 1))
        state = env.add_item(item_data, lah)
        if state is None:
            continue

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            continue

        s_tensor = {
            'occupancy_multiscale': {
                'quarter': torch.FloatTensor(state['occupancy_multiscale']['quarter']).unsqueeze(0)
            },
            'candidates':      torch.FloatTensor(state['candidates']).unsqueeze(0),
            'current_item':    torch.FloatTensor(state['current_item']).unsqueeze(0),
            'remaining_items': torch.FloatTensor(state['remaining_items']).unsqueeze(0),
            'global_features': torch.FloatTensor(state['global_features']).unsqueeze(0),
            'height_map':      torch.FloatTensor(state['height_map']).unsqueeze(0),
            'lookahead_items': torch.FloatTensor(state['lookahead_items']).unsqueeze(0),
        }

        feats = []
        for pos, rot in valid_actions:
            x, y, z = pos
            feats.append([x / max(L - 1, 1),
                          y / max(W - 1, 1),
                          z / max(H - 1, 1),
                          rot / 5.0])
        action_feats = torch.FloatTensor(feats)

        with torch.no_grad():
            fused  = net.encode_state(s_tensor)
            scores = net.score_actions(fused, action_feats)

        best_idx = scores.argmax().item()
        env.step(valid_actions[best_idx])

    return env.container


def _yz_polygon(layers):
    """
    根据分层定义构建集装器 Y-Z 截面多边形顶点（顺时针）。

    飘板型（AKE/pmc_F_ld）：下层窄、上层宽，右侧出现向外台阶。
    收角型（pmc_md11f_md/pge_md11f_md）：下层宽、上层窄，两侧出现向内台阶。
    返回 [(y, z), ...] 的封闭多边形顶点列表。
    """
    layers = sorted(layers, key=lambda l: l[0])
    n = len(layers)
    pts = []

    # 底边：从左(y_min)到右(y_max)
    pts.append((layers[0][2], layers[0][0]))
    pts.append((layers[0][3], layers[0][0]))

    # 右侧壁：由下往上，遇到 y_max 变化时插入水平台阶
    for i in range(n):
        z_start, z_end, _, y_max = layers[i]
        if i > 0 and layers[i - 1][3] != y_max:
            pts.append((y_max, z_start))   # 水平台阶
        pts.append((y_max, z_end))

    # 顶边：从右(y_max)到左(y_min)
    pts.append((layers[-1][2], layers[-1][1]))

    # 左侧壁：由上往下，遇到 y_min 变化时插入水平台阶
    for i in range(n - 1, -1, -1):
        z_start, z_end, y_min, _ = layers[i]
        if i < n - 1 and layers[i + 1][2] != y_min:
            pts.append((y_min, z_end))     # 水平台阶
        if i > 0:
            pts.append((y_min, z_start))

    return pts


def _save_loading_diagram(container, ratio_pct, ratio_dir):
    """将容器装载结果渲染为 3D 装载图并保存到 ratio_dir。"""
    if container is None or not container.items:
        print(f"  [装载图] 无已放置货物，跳过")
        return

    L, W, H   = container.L, container.W, container.H
    num_items = len(container.items)
    cfg       = IRREGULAR_CONTAINER_CONFIGS[container.container_type]

    fig = plt.figure(figsize=(14, 9))
    ax  = fig.add_subplot(111, projection='3d')

    # ── 集装器实际形状（按分层多边形截面拉伸） ────────────────────────
    yz_pts     = _yz_polygon(cfg['layers'])
    np_yz      = len(yz_pts)
    front_face = [[L, y, z] for y, z in yz_pts]   # x = L 面
    back_face  = [[0, y, z] for y, z in yz_pts]   # x = 0 面
    side_faces = []
    for i in range(np_yz):
        y1, z1 = yz_pts[i]
        y2, z2 = yz_pts[(i + 1) % np_yz]
        side_faces.append([[0, y1, z1], [L, y1, z1], [L, y2, z2], [0, y2, z2]])

    ax.add_collection3d(Poly3DCollection(
        [front_face, back_face] + side_faces,
        alpha=0.07, facecolor='lightsteelblue',
        edgecolor='steelblue', linewidths=1.0))

    # ── 货物 ──────────────────────────────────────────────────────────
    try:
        cmap = plt.colormaps['tab20']
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(num_items)]

    for idx, item in enumerate(container.items):
        if item.position is None:
            continue
        x, y, z = item.position
        l, w, h = item.get_current_dims()
        verts = np.array([
            [x,   y,   z],   [x+l, y,   z],   [x+l, y+w, z],   [x,   y+w, z],
            [x,   y,   z+h], [x+l, y,   z+h], [x+l, y+w, z+h], [x,   y+w, z+h],
        ])
        faces = [
            [verts[0], verts[1], verts[2], verts[3]],
            [verts[4], verts[5], verts[6], verts[7]],
            [verts[0], verts[1], verts[5], verts[4]],
            [verts[2], verts[3], verts[7], verts[6]],
            [verts[0], verts[3], verts[7], verts[4]],
            [verts[1], verts[2], verts[6], verts[5]],
        ]
        ax.add_collection3d(Poly3DCollection(
            faces, alpha=0.75, facecolor=colors[idx],
            edgecolor='k', linewidths=0.3))

    ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
    ax.set_xlabel('X（长）'); ax.set_ylabel('Y（宽）'); ax.set_zlabel('Z（高）')
    util = container.volume_used / container.valid_volume
    ax.set_title(
        f'装载图  监督比例 {ratio_pct}%  ·  利用率 {util:.2%}  ·  已装 {num_items} 件',
        fontsize=12, fontweight='bold')
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()

    path = os.path.join(ratio_dir, 'loading_diagram.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  装载图   → {path}")


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

    # ── 子进程模式：由主进程带集装器类型参数调用 ──────────────────────
    # 例：python train.py pmc_md11f_md
    # 此时 parameter_irregular.py 已被主进程替换为该集装器的配置，
    # 直接执行当前配置集装器的全比例训练即可。
    if len(sys.argv) == 2 and sys.argv[1] in CONTAINER_TYPES:
        from test import test_excel_data_loading_corrected
        if not test_excel_data_loading_corrected():
            print("数据加载失败，请检查 Excel 文件和配置")
            sys.exit(1)

        if not ensure_save_directory(SAVE_DIR):
            print("无法创建保存目录，训练终止")
            sys.exit(1)

        try:
            episode_assignments, excel_data = generate_training_episodes_from_excel(
                EXCEL_FILE, DATA_SCALE, NUM_PROCESSES
            )
            all_episode_ids = [ep for eps in episode_assignments for ep in eps]
            print(f"\n数据加载完成：共 {len(all_episode_ids)} 个 episodes")
        except Exception as e:
            print(f"读取 Excel 数据失败: {e}")
            sys.exit(1)

        mp.set_start_method('spawn', force=True)

        ratios    = [round(i / 10, 1) for i in range(11)]
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
        sys.exit(0)

    # ── 主进程模式：依次为四类集装器启动子进程 ─────────────────────────
    import subprocess

    print("A3C 三维装箱训练  —  四类集装器 × 0%~100% 监督比例对比实验")
    print("=" * 60)
    print(f"  集装器列表: {CONTAINER_TYPES}")
    print(f"  监督比例  : 0%, 10%, ..., 100%（共 11 组）")
    print("=" * 60)

    original_content = _read_param_file()
    all_container_stats = {}   # {container_type: stats_or_None}
    t_total = time.time()

    try:
        for ctype in CONTAINER_TYPES:
            print(f"\n{'═'*60}")
            print(f"  开始训练集装器: {ctype}")
            print(f"{'═'*60}")

            patched = _patch_container_type(original_content, ctype)
            _write_param_file(patched)

            proc = subprocess.run(
                [sys.executable, __file__, ctype],
                cwd=_SCRIPT_DIR,
            )

            if proc.returncode != 0:
                print(f"  [{ctype}] 子进程异常退出 (code={proc.returncode})")
                all_container_stats[ctype] = None
                continue

            # 从该集装器的汇总文件中读回最优比例的结果，用于跨容器汇总
            summary_txt = os.path.join(
                _SCRIPT_DIR, f'models_irregular_{ctype}', 'all_ratios_summary.txt'
            )
            cstats = {'container_type': ctype}
            if os.path.exists(summary_txt):
                with open(summary_txt, encoding='utf-8') as f:
                    txt = f.read()
                # 取装载率最高的那一行
                best_util = 0.0
                for line in txt.splitlines():
                    m = re.search(r'(\d+\.\d+)%', line)
                    if m:
                        val = float(m.group(1)) / 100.0
                        if val > best_util:
                            best_util = val
                            cstats['avg_utilization'] = val
                            # 尝试提取同行的其他字段
                            nums = re.findall(r'[-\d]+\.\d+', line)
                            if len(nums) >= 3:
                                cstats['avg_reward_last50'] = float(nums[1])
                                cstats['max_reward']        = float(nums[2])
            all_container_stats[ctype] = cstats

    finally:
        _write_param_file(original_content)
        print(f"\n  parameter_irregular.py 已恢复原始配置（{CONTAINER_TYPE}）")

    elapsed    = int(time.time() - t_total)
    h, rem     = divmod(elapsed, 3600)
    m_tot, sec = divmod(rem, 60)
    print(f"\n  全部集装器训练完成，总耗时 {h}h {m_tot}m {sec}s")

    # 打印跨容器最终汇总
    valid = [s for s in all_container_stats.values() if s is not None]
    if valid:
        sep = "=" * 70
        print(f"\n{sep}")
        print("  四类集装器  全监督比例  最优装载率汇总")
        print(sep)
        for s in valid:
            util_str = f"{s.get('avg_utilization', 0)*100:.2f}%" if 'avg_utilization' in s else 'N/A'
            print(f"  {s['container_type']:<20} 最优装载率: {util_str}")
        best = max(valid, key=lambda s: s.get('avg_utilization', 0))
        print(sep)
        print(f"  最优集装器 → {best['container_type']}  "
              f"（装载率 {best.get('avg_utilization', 0)*100:.2f}%）")
