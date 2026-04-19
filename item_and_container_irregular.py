"""
物品和容器类（不规则集装器版本）
在原 item_and_container.py 基础上增加：
  1. 轮廓掩码 M（三维bool数组，标记合法可用体素）
  2. 合法性检查增加轮廓约束
  3. volume_capacity 改为 valid_volume（合法可用体积）
  4. 其余逻辑与原版完全一致
"""
import numpy as np
from typing import List, Tuple

from my_imports import *
from parameter_irregular import IRREGULAR_CONTAINER_CONFIGS, CONTAINER_TYPE


# ======================================================================
# 轮廓掩码生成函数
# ======================================================================
def build_contour_mask(container_type: str) -> np.ndarray:
    """
    根据集装器类型构建三维轮廓掩码 M。
    M[x, y, z] = True  → 该体素位于合法装载区域
    M[x, y, z] = False → 该体素位于轮廓约束禁区

    分层定义：每层指定 z 方向范围和 y 方向合法范围，
    x 方向（长度方向）无约束，全部合法。
    """
    cfg = IRREGULAR_CONTAINER_CONFIGS[container_type]
    L, W, H = cfg['size']
    layers  = cfg['layers']

    M = np.zeros((L, W, H), dtype=bool)

    for (z_start, z_end, y_min, y_max) in layers:
        # x方向全部合法，y方向受约束，z方向按分层范围
        M[:, y_min:y_max, z_start:z_end] = True

    return M


# ======================================================================
# MetaBox（与原版完全相同）
# ======================================================================
class MetaBox:
    def __init__(self, x, y, z, lx, ly, lz):
        self.x  = x
        self.y  = y
        self.z  = z
        self.lx = lx
        self.ly = ly
        self.lz = lz

    def split(self, divide_flag, pos):
        if divide_flag == 0:
            box1 = MetaBox(pos, self.y, self.z, self.lx, self.ly, self.lz)
            box2 = MetaBox(self.x - pos, self.y, self.z, self.lx + pos, self.ly, self.lz)
        elif divide_flag == 1:
            box1 = MetaBox(self.x, pos, self.z, self.lx, self.ly, self.lz)
            box2 = MetaBox(self.x, self.y - pos, self.z, self.lx, self.ly + pos, self.lz)
        elif divide_flag == 2:
            box1 = MetaBox(self.x, self.y, pos, self.lx, self.ly, self.lz)
            box2 = MetaBox(self.x, self.y, self.z - pos, self.lx, self.ly, self.lz + pos)
        return box1, box2

    def __str__(self):
        return '(%d,%d,%d,%d,%d,%d)' % (self.x, self.y, self.z,
                                          self.lx, self.ly, self.lz)


# ======================================================================
# Item（与原版完全相同）
# ======================================================================
class Item:
    def __init__(self, l, w, h):
        self.dims     = (l, w, h)
        self.volume   = l * w * h
        self.position = None
        self.rotation = 0

    def get_current_dims(self):
        l, w, h = self.dims
        rotations = [
            (l, w, h), (l, h, w), (w, l, h),
            (w, h, l), (h, l, w), (h, w, l)
        ]
        return rotations[self.rotation]

    def copy(self):
        new_item          = Item(*self.dims)
        new_item.rotation = self.rotation
        new_item.position = self.position
        return new_item


# ======================================================================
# IrregularContainer（在原 ImprovedContainer 基础上扩展）
# ======================================================================
class IrregularContainer:
    """
    不规则集装器。
    与 ImprovedContainer 接口完全兼容，额外提供：
      - self.contour_mask  : 三维轮廓掩码 M
      - self.valid_volume  : 合法可用体积（掩码内体素数）
      - volume_capacity    : 同 valid_volume（向后兼容原代码中的 volume_capacity 引用）
    """

    def __init__(self, container_type: str = None):
        if container_type is None:
            container_type = CONTAINER_TYPE

        cfg             = IRREGULAR_CONTAINER_CONFIGS[container_type]
        self.L, self.W, self.H = cfg['size']
        self.container_type    = container_type
        self.type_id           = cfg['type_id']

        # ── 核心新增：轮廓掩码 ──────────────────────────────────────
        self.contour_mask  = build_contour_mask(container_type)   # (L,W,H) bool
        self.valid_volume  = int(np.sum(self.contour_mask))
        # ────────────────────────────────────────────────────────────

        # 向后兼容：原代码中引用 volume_capacity 的地方均可直接使用
        self.volume_capacity = self.valid_volume

        # 以下与原版完全相同
        self.items                = []
        self.occupancy            = np.zeros((self.L, self.W, self.H), dtype=bool)
        self.center_of_gravity    = (0, 0, 0)
        self.volume_used          = 0
        self.plain                = np.zeros((self.L, self.W), dtype=np.int32)
        self.meta_list            = [MetaBox(self.L, self.W, self.H, 0, 0, 0)]
        self.candidates           = []
        self.placement_attempts   = 0
        self.successful_placements = 0
        self._add_candidate()

    # ──────────────────────────────────────────────────────────────────
    # 以下方法与原版完全相同（不改动）
    # ──────────────────────────────────────────────────────────────────

    def _add_candidate(self):
        new_list = []
        for i in range(len(self.meta_list)):
            mb = self.meta_list[i]
            max_x = min(mb.lx + mb.x, self.L)
            max_y = min(mb.ly + mb.y, self.W)

            if mb.lx >= self.L or mb.ly >= self.W or max_x <= 0 or max_y <= 0:
                continue

            start_x = max(0, mb.lx)
            start_y = max(0, mb.ly)

            occupied_area = self.plain[start_x:max_x, start_y:max_y]
            area_size     = (max_x - start_x) * (max_y - start_y)

            if area_size <= 0:
                continue

            check_sum = np.sum(occupied_area == mb.lz)
            if check_sum == area_size:
                self.candidates.append(mb)
            else:
                new_list.append(mb)

        self.meta_list = new_list

    def _update_plain(self, box):
        self.plain[box.lx:box.lx + box.x, box.ly:box.ly + box.y] += box.z

    def _calculate_max_area(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        if (x >= self.L or y >= self.W or z >= self.H or
                x < 0 or y < 0 or z < 0):
            return 0, 0, 0
        if self.occupancy[x, y, z]:
            return 0, 0, 0

        max_volume = 0
        best_dims  = (0, 0, 0)

        max_possible_length = self.L - x
        max_possible_width  = self.W - y
        max_possible_height = self.H - z

        for length in range(1, max_possible_length + 1):
            x_blocked = False
            for j in range(y, y + 1):
                for k in range(z, z + 1):
                    if self.occupancy[x + length - 1, j, k]:
                        x_blocked = True
                        break
                if x_blocked:
                    break
            if x_blocked:
                break

            for width in range(1, max_possible_width + 1):
                y_blocked = False
                for i in range(x, x + length):
                    for k in range(z, z + 1):
                        if self.occupancy[i, y + width - 1, k]:
                            y_blocked = True
                            break
                    if y_blocked:
                        break
                if y_blocked:
                    break

                for height in range(1, max_possible_height + 1):
                    z_blocked = False
                    for i in range(x, x + length):
                        for j in range(y, y + width):
                            if self.occupancy[i, j, z + height - 1]:
                                z_blocked = True
                                break
                        if z_blocked:
                            break
                    if z_blocked:
                        break

                    volume = length * width * height
                    if volume > max_volume:
                        max_volume = volume
                        best_dims  = (length, width, height)

        return best_dims

    def _is_space_subset(self, space_a: MetaBox, space_b: MetaBox) -> bool:
        a_x1, a_y1, a_z1 = space_a.lx, space_a.ly, space_a.lz
        a_x2 = a_x1 + space_a.x
        a_y2 = a_y1 + space_a.y
        a_z2 = a_z1 + space_a.z

        b_x1, b_y1, b_z1 = space_b.lx, space_b.ly, space_b.lz
        b_x2 = b_x1 + space_b.x
        b_y2 = b_y1 + space_b.y
        b_z2 = b_z1 + space_b.z

        return (b_x1 <= a_x1 and a_x2 <= b_x2 and
                b_y1 <= a_y1 and a_y2 <= b_y2 and
                b_z1 <= a_z1 and a_z2 <= b_z2)

    def _remove_subset_spaces(self):
        spaces_to_remove = set()
        for i in range(len(self.meta_list)):
            if i in spaces_to_remove:
                continue
            for j in range(len(self.meta_list)):
                if i != j and j not in spaces_to_remove:
                    if self._is_space_subset(self.meta_list[i], self.meta_list[j]):
                        spaces_to_remove.add(i)
                        break
        for index in sorted(spaces_to_remove, reverse=True):
            del self.meta_list[index]

    def _update_existing_spaces(self):
        i = 0
        while i < len(self.meta_list):
            mb  = self.meta_list[i]
            lwh = self._calculate_max_area(mb.lx, mb.ly, mb.lz)
            if lwh[0] == 0 and lwh[1] == 0 and lwh[2] == 0:
                del self.meta_list[i]
            else:
                mb.x = lwh[0]
                mb.y = lwh[1]
                mb.z = lwh[2]
                i += 1

    def _split_box(self, placed_box, item_dims):
        l, w, h = item_dims
        x, y, z = placed_box.lx, placed_box.ly, placed_box.lz

        if placed_box.z >= h:
            lwh = self._calculate_max_area(x, y, z + h)
            if lwh[0] != 0 and lwh[1] != 0 and lwh[2] != 0:
                self.meta_list.append(MetaBox(lwh[0], lwh[1], lwh[2], x, y, z + h))

        if placed_box.y >= w:
            lwh = self._calculate_max_area(x, y + w, z)
            if lwh[0] != 0 and lwh[1] != 0 and lwh[2] != 0:
                self.meta_list.append(MetaBox(lwh[0], lwh[1], lwh[2], x, y + w, z))

        if placed_box.x >= l:
            lwh = self._calculate_max_area(x + l, y, z)
            if lwh[0] != 0 and lwh[1] != 0 and lwh[2] != 0:
                self.meta_list.append(MetaBox(lwh[0], lwh[1], lwh[2], x + l, y, z))

        self._update_existing_spaces()
        self._remove_subset_spaces()
        self._add_candidate()

    def _calculate_support_details(self, x, y, z, l, w, h):
        bottom_support = self.occupancy[x:x + l, y:y + w, z - 1]
        support_area   = np.sum(bottom_support)
        total_area     = l * w
        support_ratio  = support_area / total_area

        corners = [(0, 0), (l - 1, 0), (0, w - 1), (l - 1, w - 1)]
        corner_support = sum(
            1 for dx, dy in corners if self.occupancy[x + dx, y + dy, z - 1]
        )

        return support_ratio, corner_support

    # ──────────────────────────────────────────────────────────────────
    # 关键改动1：is_valid_position 增加轮廓约束检查
    # ──────────────────────────────────────────────────────────────────
    def is_valid_position(self, item, position: Tuple[int, int, int],
                           rotation: int) -> bool:
        self.placement_attempts += 1

        item.rotation = rotation
        l, w, h = item.get_current_dims()
        x, y, z = position

        # 1. 边界检查（与原版相同）
        if (x < 0 or y < 0 or z < 0 or
                x + l > self.L or y + w > self.W or z + h > self.H):
            return False

        # 2. 碰撞检查（与原版相同）
        if np.any(self.occupancy[x:x + l, y:y + w, z:z + h]):
            return False

        # ── 新增：轮廓掩码约束检查 ──────────────────────────────────
        # 货物占用的所有体素必须全部位于合法装载区域内
        # 即 contour_mask 在该区域内的值全部为 True
        if not np.all(self.contour_mask[x:x + l, y:y + w, z:z + h]):
            return False
        # ─────────────────────────────────────────────────────────────

        # 3. 支撑检查（与原版相同）
        if z > 0:
            support_ratio, corner_support = self._calculate_support_details(
                x, y, z, l, w, h)
            is_supported = (
                support_ratio > 0.8 or
                (support_ratio > 0.6 and corner_support >= 2) or
                (support_ratio > 0.4 and corner_support >= 3) or
                (support_ratio > 0.2 and corner_support == 4)
            )
            if not is_supported:
                return False

        self.successful_placements += 1
        return True

    def try_place_item(self, item: Item, position: Tuple[int, int, int],
                        rotation: int) -> bool:
        original_rotation = item.rotation

        if not self.is_valid_position(item, position, rotation):
            item.rotation = original_rotation
            return False

        item.rotation = rotation
        l, w, h = item.get_current_dims()
        x, y, z = position

        self.occupancy[x:x + l, y:y + w, z:z + h] = True
        item.position = position
        self.items.append(item)
        self.volume_used += item.volume

        # 更新重心（与原版相同）
        if len(self.items) == 1:
            self.center_of_gravity = (x + l / 2, y + w / 2, z + h / 2)
        else:
            total_volume  = self.volume_used
            current_cog   = np.array(self.center_of_gravity)
            previous_vol  = total_volume - item.volume
            item_center   = np.array([x + l / 2, y + w / 2, z + h / 2])
            new_cog       = (current_cog * previous_vol + item_center * item.volume) / total_volume
            self.center_of_gravity = tuple(new_cog)

        # 更新 MetaBox（与原版相同）
        used_box_idx = None
        for idx, box in enumerate(self.candidates):
            if (box.lx <= x < box.lx + box.x and
                    box.ly <= y < box.ly + box.y and
                    box.lz <= z < box.lz + box.z):
                if (x + l <= box.lx + box.x and
                        y + w <= box.ly + box.y and
                        z + h <= box.lz + box.z):
                    used_box_idx = idx
                    break

        if used_box_idx is not None:
            used_box = self.candidates.pop(used_box_idx)
            self._update_plain(MetaBox(l, w, h, x, y, z))
            self._split_box(used_box, (l, w, h))
        else:
            self._update_plain(MetaBox(l, w, h, x, y, z))
            self._update_existing_spaces()
            self._remove_subset_spaces()
            self._add_candidate()

        # 体积验证（与原版相同）
        space_occupied_volume = np.count_nonzero(self.occupancy)
        space_items_volume    = sum(it.volume for it in self.items)
        if abs(space_occupied_volume - space_items_volume) > 0.1:
            print(f'警告：体积不匹配！占用={space_occupied_volume}，物品={space_items_volume}')
            self.volume_used = space_occupied_volume

        return True

    # ──────────────────────────────────────────────────────────────────
    # 关键改动2：get_valid_positions 候选位置预过滤轮廓约束
    # ──────────────────────────────────────────────────────────────────
    def get_valid_positions(self, item: Item) -> List[Tuple[Tuple[int, int, int], int]]:
        valid_actions = []

        for box in self.candidates:
            for rotation in range(6):
                item_copy          = item.copy()
                item_copy.rotation = rotation
                l, w, h = item_copy.get_current_dims()

                if l > box.x or w > box.y or h > box.z:
                    continue

                position = (box.lx, box.ly, box.lz)

                # 预过滤：快速检查候选位置是否在合法区域内
                # 避免进入 is_valid_position 的完整流程
                x, y, z = position
                if (x + l > self.L or y + w > self.W or z + h > self.H):
                    continue
                if not np.all(self.contour_mask[x:x + l, y:y + w, z:z + h]):
                    continue

                if self.is_valid_position(item_copy, position, rotation):
                    valid_actions.append((position, rotation))

        if not valid_actions:
            for rotation in range(6):
                item_copy          = item.copy()
                item_copy.rotation = rotation
                l, w, h = item_copy.get_current_dims()

                for x in range(max(0, self.L - l + 1)):
                    for y in range(max(0, self.W - w + 1)):
                        for z in range(max(0, self.H - h + 1)):
                            # 预过滤轮廓约束
                            if not np.all(self.contour_mask[x:x + l, y:y + w, z:z + h]):
                                continue
                            if self.is_valid_position(item_copy, (x, y, z), rotation):
                                valid_actions.append(((x, y, z), rotation))

        return valid_actions
