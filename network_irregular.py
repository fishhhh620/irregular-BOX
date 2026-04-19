"""
不规则集装器 A3C 网络
在原 network.py 基础上唯一改动：
  global_encoder 输入维度从 5 改为 9
  （新增：重心安全偏差1维 + 集装器类型one-hot 4维 - 原版COG已有3维利用率已有1维件数已有1维）
  其余结构与原版完全一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from my_imports import *
from parameter_irregular import K_MAX


class IrregularA3CNet(nn.Module):
    def __init__(self, container_size):
        super(IrregularA3CNet, self).__init__()
        self.container_size = container_size

        # 3D卷积编码器（与原版相同）
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )

        # 2D高度图编码器（与原版相同）
        self.conv2d_height = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3))
        )

        # 特征编码器（与原版相同）
        self.candidate_encoder = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU()
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU()
        )
        self.remaining_encoder = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU()
        )
        self.lookahead_encoder = nn.Sequential(
            nn.Linear(4 * K_MAX, 64), nn.ReLU()
        )

        # ── 唯一改动：输入维度 5 → 9 ──────────────────────────────
        # 原版：利用率1 + 件数1 + CG_xyz 3 = 5维
        # 新版：利用率1 + 件数1 + CG_xyz 3 + CG偏差1 + 类型one-hot 4 = 10维
        # 注：state_encoder_irregular 输出9维（含件数归一化）实际是9维
        self.global_encoder = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU()
        )
        # ─────────────────────────────────────────────────────────────

        # 特征融合层（与原版相同，总维度不变）
        total_features = (32 * 2 * 2 * 2 +   # 3D卷积
                          32 * 3 * 3 +         # 高度图
                          32 + 32 + 32 + 32 + 64)  # 其他特征 + lookahead

        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Actor/Critic 头（与原版完全相同）
        self.action_encoder = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU()
        )
        self.actor_head = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def encode_state(self, state_dict):
        batch_size = state_dict['occupancy_multiscale']['quarter'].shape[0]

        occupancy   = state_dict['occupancy_multiscale']['quarter'].unsqueeze(1)
        conv_feat   = self.conv3d(occupancy).view(batch_size, -1)

        height_map  = state_dict['height_map'].unsqueeze(1)
        height_feat = self.conv2d_height(height_map).view(batch_size, -1)

        candidate_feat  = self.candidate_encoder(state_dict['candidates'])
        item_feat       = self.item_encoder(state_dict['current_item'])
        remaining_feat  = self.remaining_encoder(state_dict['remaining_items'])
        lookahead_feat  = self.lookahead_encoder(state_dict['lookahead_items'])
        global_feat     = self.global_encoder(state_dict['global_features'])  # 9维输入

        all_features = torch.cat([
            conv_feat, height_feat, candidate_feat,
            item_feat, remaining_feat, lookahead_feat, global_feat
        ], dim=1)

        return self.feature_fusion(all_features)

    def score_actions(self, fused_features, action_feats):
        N          = action_feats.shape[0]
        action_enc = self.action_encoder(action_feats)
        state_exp  = fused_features.expand(N, -1)
        combined   = torch.cat([state_exp, action_enc], dim=1)
        return self.actor_head(combined).squeeze(-1)

    def forward(self, state_dict):
        fused_features = self.encode_state(state_dict)
        state_value    = self.critic_head(fused_features)
        return None, state_value
