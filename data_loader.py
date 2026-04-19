"""
Excel数据加载函数

Excel 列结构（共11列，索引0-10）：
  0: 数据集编号
  1: 货物序号
  2: 长（最小2，最大5）
  3: 宽（最小2，最大5）
  4: 高（最小2，最大5）
  5: 起始X（箱子长10）
  6: 起始Y（箱子宽10）
  7: 起始Z（箱子高10）
  8: 选择货物的编号（专家动作1）
  9: 货物选择空闲空间的编号（专家动作2）
 10: 空闲空间的最大值（专家动作3）
"""
import numpy as np
import pandas as pd
import os

from my_imports import *
from parameter_irregular import EXCEL_FILE, DATA_SCALE


def ensure_save_directory(save_dir):
    """确保保存目录存在"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"✅ 确保保存目录存在: {save_dir}")
        return True
    except Exception as e:
        print(f"❌ 创建保存目录失败: {str(e)}")
        return False


def load_excel_data(excel_file, data_scale):
    """
    从Excel文件读取数据。
    列结构：数据集编号、货物序号、长、宽、高、起始X、起始Y、起始Z、
            选择货物编号、空闲空间编号、空闲空间最大值
    """
    try:
        sheet_name = f"规模{data_scale}"
        print(f"正在读取Excel文件: {excel_file}, 工作表: {sheet_name}")

        data = pd.read_excel(excel_file, sheet_name=sheet_name)
        data = np.array(data)

        print(f"成功读取数据，共 {data.shape[0]} 行，{data.shape[1]} 列")
        print(f"列说明: 数据集编号(0) 货物序号(1) 长(2) 宽(3) 高(4) "
              f"起始X(5) 起始Y(6) 起始Z(7) 货物编号(8) 空间编号(9) 空间最大值(10)")
        return data
    except Exception as e:
        print(f"读取Excel文件失败: {str(e)}")
        raise e


def extract_episode_data_corrected(data, episode_id):
    """
    提取单个 episode 的所有数据。

    返回:
        items_data     : [(长, 宽, 高), ...]
        positions_data : [(x, y, z), ...]       — 专家放置位置
        action_targets : [(item_id, space_id, space_max), ...]  — 专家动作目标
    """
    episode_mask = data[:, 0] == episode_id
    episode_data = data[episode_mask]

    if len(episode_data) == 0:
        print(f"警告：episode {episode_id} 没有数据")
        return [], [], []

    items_data = []
    positions_data = []
    action_targets = []

    for row in episode_data:
        # 货物尺寸（列2-4）
        length = int(row[2])
        width  = int(row[3])
        height = int(row[4])
        items_data.append((length, width, height))

        # 专家放置位置（列5-7）
        if data.shape[1] > 7:
            x = int(row[5])
            y = int(row[6])
            z = int(row[7])
            positions_data.append((x, y, z))

        # 专家动作目标（列8-10）
        if data.shape[1] > 10:
            item_action  = int(row[8])   # 选择货物的编号
            space_action = int(row[9])   # 空闲空间的编号
            space_max    = int(row[10])  # 空闲空间的最大值
            action_targets.append((item_action, space_action, space_max))

    return items_data, positions_data, action_targets


class OnlineItemIterator:
    """
    Online模式：逐个返回货物的迭代器。
    """
    def __init__(self, episode_data, optimal_placements, action_targets=None):
        self.episode_data      = episode_data
        self.optimal_placements = optimal_placements
        self.action_targets    = action_targets or []
        self.current_idx       = 0
        self.total_items       = len(episode_data)

    def has_next(self):
        return self.current_idx < self.total_items

    def get_next_item(self):
        """
        返回: (item_data, optimal_pos, action_target)
          - optimal_pos   : (x, y, z) 或 None
          - action_target : (item_id, space_id, space_max) 或 None
        """
        if not self.has_next():
            return None

        item_data     = self.episode_data[self.current_idx]
        optimal_pos   = (self.optimal_placements[self.current_idx]
                         if self.current_idx < len(self.optimal_placements) else None)
        action_target = (self.action_targets[self.current_idx]
                         if self.current_idx < len(self.action_targets) else None)

        self.current_idx += 1
        return item_data, optimal_pos, action_target

    def reset(self):
        self.current_idx = 0

    def get_progress(self):
        return self.current_idx, self.total_items

    def peek_next_items(self, n):
        end = min(self.current_idx + n, self.total_items)
        return self.episode_data[self.current_idx:end]


def get_labeled_episode_ids(all_episode_ids, supervised_ratio):
    """
    按 supervised_ratio 比例取前 N 个 episode 作为有标签数据集（保持原始顺序）。
    supervised_ratio=0.0 → 全部无标签（纯 RL）
    supervised_ratio=1.0 → 全部有标签（全监督）
    """
    if supervised_ratio <= 0.0:
        return set()
    if supervised_ratio >= 1.0:
        labeled_ids = set(all_episode_ids)
    else:
        n_labeled   = max(1, int(len(all_episode_ids) * supervised_ratio))
        labeled_ids = set(all_episode_ids[:n_labeled])
    print(f"数据集划分：共 {len(all_episode_ids)} 个episodes，"
          f"有标签（前 {len(labeled_ids)} 个）({supervised_ratio*100:.0f}%)，"
          f"无标签 {len(all_episode_ids) - len(labeled_ids)} 个")
    return labeled_ids


def generate_training_episodes_from_excel(excel_file, data_scale, num_workers=4):
    """
    为每个 worker 分配 episode IDs。
    返回: (episode_assignments, data)
    """
    data = load_excel_data(excel_file, data_scale)
    unique_episodes  = np.unique(data[:, 0])
    max_episodes     = len(unique_episodes)

    print(f"发现 {max_episodes} 个不同的episode: {unique_episodes[:10]}...")

    episode_assignments  = []
    episodes_per_worker  = max(1, max_episodes // num_workers)

    for worker_id in range(num_workers):
        start_idx = worker_id * episodes_per_worker
        end_idx   = min((worker_id + 1) * episodes_per_worker, max_episodes)
        if worker_id == num_workers - 1:
            end_idx = max_episodes

        worker_episode_ids = unique_episodes[start_idx:end_idx].tolist()
        episode_assignments.append(worker_episode_ids)
        print(f"Worker {worker_id} 分配episodes: {worker_episode_ids[:5]}... (共{len(worker_episode_ids)}个)")

    return episode_assignments, data
