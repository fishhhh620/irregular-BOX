"""
奖励计算器
"""
import numpy as np

from item_and_container_irregular import Item, IrregularContainer as ImprovedContainer


class RewardCalculator:
    def __init__(self):
        self.prev_utilization = 0.0
        self.placement_history = []

    def calculate_reward(self, container: ImprovedContainer, item: Item, action_success: bool):
        if not action_success:
            return -5.0

        reward = 0.0

        # 1. 体积利用率增量奖励 (核心奖励)
        current_utilization = container.volume_used / container.volume_capacity
        utilization_improvement = current_utilization - self.prev_utilization
        reward += 15.0 * utilization_improvement
        self.prev_utilization = current_utilization

        # 2. 高度惩罚 (保留)
        height_penalty = item.position[2] / container.H
        reward -= 2.0 * height_penalty

        # 3. 稳定性奖励 (保留)
        stability_bonus = self._calculate_stability_bonus(container, item)
        reward += 1.0 * stability_bonus

        return reward

    def _calculate_stability_bonus(self, container: ImprovedContainer, item: Item):
        x, y, z = item.position
        l, w, h = item.get_current_dims()

        if z == 0:
            return 1.0

        support_area = np.sum(container.occupancy[x:x + l, y:y + w, z - 1])
        total_area = l * w
        support_ratio = support_area / total_area

        return support_ratio