"""
不规则集装器状态编码器
在原 state_encoder.py 基础上改动两处：
  1. global_features 增加重心安全偏差 |CG_y - CG_safe| / W（第三章式3-63）
  2. encode_state 输出中增加 contour_mask 编码（供网络使用）
  3. _encode_global_features 输出从5维扩展为9维（加入4维集装器类型one-hot）
"""
import numpy as np
import torch

from item_and_container_irregular import Item, IrregularContainer
from my_imports import *
from parameter_irregular import K_MAX, IRREGULAR_CONTAINER_CONFIGS


class IrregularStateEncoder:
    def __init__(self, container_type: str):
        cfg             = IRREGULAR_CONTAINER_CONFIGS[container_type]
        self.L, self.W, self.H = cfg['size']
        self.container_type    = container_type
        self.type_id           = cfg['type_id']
        self.valid_volume      = cfg['valid_volume']
        self.container_size    = cfg['size']

        # 重心安全中心值（对称集装器取W/2，AKE非对称取轮廓约束修正值）
        # 对称集装器：pmc_F_ld / pmc_md11f_md / pge_md11f_md → W/2
        # AKE（非对称飘板）：取轮廓约束下层y方向中心 ≈ 7.5（离散）
        if container_type == 'AKE':
            self.cg_safe_y = 7.5 / self.W   # 归一化后的重心安全中心
        else:
            self.cg_safe_y = 0.5             # W/2 归一化后为0.5

    def encode_state(self, container: IrregularContainer,
                      current_item: Item,
                      remaining_items: List[Item],
                      lookahead_items: List[Item] = None):
        occupancy_features = self._encode_occupancy_multiscale(container)
        candidate_features = self._encode_candidates(container)
        item_features      = self._encode_current_item(current_item)
        remaining_features = self._encode_remaining_items(remaining_items)
        global_features    = self._encode_global_features(container)   # 现在是9维
        height_map         = self._create_height_map(container)
        lookahead_features = self._encode_lookahead_items(lookahead_items or [])

        return {
            'occupancy_multiscale': occupancy_features,
            'candidates':           candidate_features,
            'current_item':         item_features,
            'remaining_items':      remaining_features,
            'global_features':      global_features,
            'height_map':           height_map,
            'lookahead_items':      lookahead_features,
        }

    def _encode_occupancy_multiscale(self, container: IrregularContainer):
        occupancy     = container.occupancy.astype(np.float32)
        quarter_scale = self._downsample_3d(occupancy, 2)
        return {'quarter': quarter_scale}

    def _downsample_3d(self, array, factor):
        L, W, H = array.shape
        new_L = max(1, L // factor)
        new_W = max(1, W // factor)
        new_H = max(1, H // factor)
        downsampled = np.zeros((new_L, new_W, new_H))
        for i in range(new_L):
            for j in range(new_W):
                for k in range(new_H):
                    xs, xe = i * factor, min((i + 1) * factor, L)
                    ys, ye = j * factor, min((j + 1) * factor, W)
                    zs, ze = k * factor, min((k + 1) * factor, H)
                    region = array[xs:xe, ys:ye, zs:ze]
                    downsampled[i, j, k] = np.mean(region)
        return downsampled

    def _encode_candidates(self, container: IrregularContainer):
        if not container.candidates:
            return np.zeros(8, dtype=np.float32)

        volumes   = [box.x * box.y * box.z for box in container.candidates]
        positions = [(box.lx, box.ly, box.lz) for box in container.candidates]

        features = [
            min(1.0, len(container.candidates) / 20.0),
            np.mean(volumes) / (self.L * self.W * self.H),
            np.max(volumes)  / (self.L * self.W * self.H),
            np.min(volumes)  / (self.L * self.W * self.H),
            np.mean([pos[2] for pos in positions]) / self.H,
            len([v for v in volumes if v >= 8])  / len(volumes),
            len([v for v in volumes if v <= 2])  / len(volumes),
            np.sum(volumes) / (self.L * self.W * self.H),
        ]
        return np.array(features, dtype=np.float32)

    def _encode_current_item(self, item: Item):
        dims            = np.array(item.dims, dtype=np.float32)
        normalized_dims = dims / max(self.container_size)
        features        = np.concatenate([
            normalized_dims,
            [item.volume / (self.L * self.W * self.H)]
        ])
        return features

    def _encode_remaining_items(self, remaining_items: List[Item]):
        if not remaining_items:
            return np.zeros(8, dtype=np.float32)

        volumes              = [item.volume for item in remaining_items]
        total_remaining_vol  = sum(volumes)
        features = [
            min(1.0, len(remaining_items) / 100.0),
            total_remaining_vol / (self.L * self.W * self.H),
            np.mean(volumes) / (self.L * self.W * self.H),
            np.max(volumes)  / (self.L * self.W * self.H),
            np.min(volumes)  / (self.L * self.W * self.H),
            len([v for v in volumes if v >= 20]) / len(volumes),
            len([v for v in volumes if v <= 8])  / len(volumes),
            0.0,
        ]
        return np.array(features, dtype=np.float32)

    def _encode_global_features(self, container: IrregularContainer):
        """
        全局特征（9维）：
          [0]   : 空间利用率 ρ（以valid_volume为分母）
          [1]   : 已放置货物数（归一化）
          [2-4] : 归一化重心坐标 CG_x/L, CG_y/W, CG_z/H
          [5]   : 重心安全偏差 |CG_y/W - CG_safe_y|（第三章新增）
          [6-8] : 集装器类型 one-hot 编码 T（4维，共索引6~9）
        """
        # 利用率：分母为 valid_volume（合法可用体积）
        utilization = container.volume_used / container.valid_volume

        cog            = np.array(container.center_of_gravity)
        normalized_cog = cog / np.array([self.L, self.W, self.H])

        # 重心安全偏差（归一化后的y方向重心与安全中心的距离）
        cg_y_norm    = normalized_cog[1] if container.volume_used > 0 else self.cg_safe_y
        cg_deviation = abs(cg_y_norm - self.cg_safe_y)

        # 集装器类型 one-hot（4维）
        type_onehot = np.zeros(4, dtype=np.float32)
        type_onehot[container.type_id] = 1.0

        features = [
            utilization,
            len(container.items) / 100.0,
        ]
        features.extend(normalized_cog)          # 3维
        features.append(cg_deviation)            # 1维（新增）
        features = np.array(features, dtype=np.float32)
        features = np.concatenate([features, type_onehot])  # 总计9维

        return features

    def _create_height_map(self, container: IrregularContainer):
        height_map = np.zeros((self.L, self.W), dtype=np.float32)
        for x in range(self.L):
            for y in range(self.W):
                column = container.occupancy[x, y, :]
                if np.any(column):
                    height_map[x, y] = np.max(np.where(column)[0]) + 1
                else:
                    height_map[x, y] = 0
        return height_map / self.H

    def _encode_lookahead_items(self, lookahead_items: List[Item]):
        feature_dim = 4
        max_items   = K_MAX
        features    = np.zeros(feature_dim * max_items, dtype=np.float32)
        for idx, item in enumerate(lookahead_items[:max_items]):
            dims      = np.array(item.dims, dtype=np.float32) / max(self.container_size)
            vol_ratio = item.volume / (self.L * self.W * self.H)
            vec       = np.concatenate([dims, [vol_ratio]]).astype(np.float32)
            start     = idx * feature_dim
            features[start:start + feature_dim] = vec
        return features
