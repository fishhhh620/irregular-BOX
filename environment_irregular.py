"""
不规则集装器装箱环境
在原 environment.py 基础上改动两处：
  1. 使用 IrregularContainer 替代 ImprovedContainer
  2. finish_episode 利用率分母改为 valid_volume（合法可用体积）
"""
import numpy as np

from item_and_container_irregular import Item, IrregularContainer
from reward_calculator import RewardCalculator
from state_encoder_irregular import IrregularStateEncoder
from parameter_irregular import CONTAINER_SIZE, CONTAINER_TYPE


class IrregularBinPackingEnv:
    """
    不规则集装器装箱环境（与原 SupervisedBinPackingEnv 接口完全兼容）
    """

    def __init__(self, container_type=None, optimal_placements=None):
        self.container_type      = container_type or CONTAINER_TYPE
        self.optimal_placements  = optimal_placements or []

        self.reward_calculator = RewardCalculator()
        self.state_encoder     = IrregularStateEncoder(self.container_type)

        self.reset()

    def reset(self):
        """重置环境"""
        # ── 改动1：使用不规则集装器 ──
        self.container      = IrregularContainer(self.container_type)
        self.current_item   = None
        self.lookahead_items = []
        self.items_placed   = 0
        self.reward_calculator = RewardCalculator()
        return None

    def _get_state(self):
        if self.current_item is None:
            return None
        return self.state_encoder.encode_state(
            self.container, self.current_item, [], self.lookahead_items
        )

    def add_item(self, item_data, lookahead_item_datas=None):
        l, w, h = item_data[:3]
        self.current_item    = Item(l, w, h)
        self.lookahead_items = []
        if lookahead_item_datas:
            for d in lookahead_item_datas:
                dl, dw, dh = d[:3]
                self.lookahead_items.append(Item(dl, dw, dh))
        return self._get_state()

    def step(self, action):
        if self.current_item is None:
            return None, 0, False

        position, rotation = action
        success = self.container.try_place_item(self.current_item, position, rotation)

        base_reward  = self.reward_calculator.calculate_reward(
            self.container, self.current_item, success
        )
        total_reward = base_reward

        if success:
            self.items_placed += 1
            self.current_item = None

        return self._get_state(), total_reward, False

    def finish_episode(self):
        """
        结束当前 episode，计算终端奖励。
        ── 改动2：分母改为 valid_volume（合法可用体积）──
        """
        final_utilization = self.container.volume_used / self.container.valid_volume
        terminal_bonus    = 50.0 * final_utilization
        return terminal_bonus

    def get_valid_actions(self):
        if self.current_item is None:
            return []
        return self.container.get_valid_positions(self.current_item)

    def get_optimal_action_hint(self):
        if self.items_placed >= len(self.optimal_placements):
            return None

        optimal_position = self.optimal_placements[self.items_placed]
        valid_actions    = self.get_valid_actions()

        if not valid_actions:
            return None

        best_action  = None
        min_distance = float('inf')

        for action in valid_actions:
            position, rotation = action
            distance = np.sqrt(
                sum((a - o) ** 2 for a, o in zip(position, optimal_position))
            )
            if distance < min_distance:
                min_distance = distance
                best_action  = action

        return best_action
