"""
模型测试脚本 - 支持预见窗口排序装载
用于加载训练好的模型，测试新数据（数据编号、长、宽、高）

新增功能：
- 支持预见窗口（lookahead window）
- 在窗口内按体积排序，优先装载大体积货物
- 动态维护k个货物的预见窗口
- 3D可视化装载结果，每个货物用不同颜色显示

使用方法:
1. 从Excel文件读取数据（推荐）:
   python test_model.py              # 测试所有episode，使用默认k=4
   python test_model.py 1             # 测试episode 1，使用默认k=4
   python test_model.py 1 --k 2       # 测试episode 1，使用k=2预见窗口
   
2. 从代码中使用:
   from test_model import test_from_excel_with_lookahead
   results = test_from_excel_with_lookahead(episode_id=1, k=2)
   
   from test_model import ModelTesterWithLookahead
   tester = ModelTesterWithLookahead(k=2)
   results = tester.test_multiple_items_with_lookahead(items)

数据格式:
- 输入数据可以是 (数据编号, 长, 宽, 高) 或 (长, 宽, 高)
- 示例: (1, 3, 3, 3) 或 (3, 3, 3)

输出结果:
每个结果包含:
- success: 是否成功放置
- data_id: 数据编号
- item_size: 物品尺寸 (长, 宽, 高)
- volume: 物品体积
- position: 放置位置 (x, y, z)
- rotation: 旋转角度
- reward: 奖励值
- container_utilization: 容器利用率
- window_info: 预见窗口信息
"""
import os
# Temporary workaround: allow duplicate OpenMP runtimes to proceed.
# Prefer resolving package-level conflicts (MKL/OpenMP) in the environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import time
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

try:
    from parameter_irregular import CONTAINER_SIZE, SAVE_MODEL_PATH, EXCEL_FILE, DATA_SCALE, LOOKAHEAD_K
    from network_irregular import IrregularA3CNet as ImprovedA3CNet
    from environment_irregular import IrregularBinPackingEnv as SupervisedBinPackingEnv
    from item_and_container_irregular import Item
    from data_loader import load_excel_data, extract_episode_data_corrected
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from parameter_irregular import CONTAINER_SIZE, SAVE_MODEL_PATH, EXCEL_FILE, DATA_SCALE, LOOKAHEAD_K
    from network_irregular import IrregularA3CNet as ImprovedA3CNet
    from environment_irregular import IrregularBinPackingEnv as SupervisedBinPackingEnv
    from item_and_container_irregular import Item
    from data_loader import load_excel_data, extract_episode_data_corrected


class ModelTesterWithLookahead:
    """支持预见窗口排序的模型测试类"""
    
    def __init__(self, model_path=None, k=None):
        """
        初始化测试器
        参数:
        model_path: 模型文件路径，如果为None则使用默认路径
        k: 预见窗口大小，如果为None则使用parameter.py中的LOOKAHEAD_K
        """
        self.container_size = CONTAINER_SIZE
        self.model_path = model_path or SAVE_MODEL_PATH
        self.k = k or LOOKAHEAD_K  # 预见窗口大小
        
        # 加载模型
        self.model = ImprovedA3CNet(self.container_size)
        self._load_model()
        self.model.eval()  # 设置为评估模式
        
        print(f"✅ 模型加载成功: {self.model_path}")
        print(f"📊 预见窗口大小: k={self.k}")
    
    def _visualize_packing_3d(self, container, title="Packing Visualization", save_path=None):
        """
        3D可视化装箱结果
        参数:
        container: ImprovedContainer对象
        title: 图表标题
        save_path: 保存路径（可选）
        """
        try:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
            L, W, H = container.L, container.W, container.H
            
            # 生成颜色列表，每个货物使用不同颜色
            num_items = len(container.items)
            if num_items == 0:
                print("⚠️ 没有已放置的货物，无法可视化")
                return
            
            # 使用colormap生成不同颜色
            try:
                # 新版本matplotlib
                cmap = plt.colormaps['tab20']
            except (AttributeError, KeyError):
                # 旧版本matplotlib
                cmap = plt.cm.get_cmap('tab20')
            colors = [cmap(i % 20) for i in range(num_items)]
            
            # 绘制每个货物
            for idx, item in enumerate(container.items):
                if item.position is None:
                    continue
                
                x, y, z = item.position
                l, w, h = item.get_current_dims()
                
                # 定义长方体的8个顶点
                vertices = np.array([
                    [x, y, z],           # 0: 左下前
                    [x + l, y, z],       # 1: 右下前
                    [x + l, y + w, z],  # 2: 右上前
                    [x, y + w, z],      # 3: 左上前
                    [x, y, z + h],      # 4: 左下后
                    [x + l, y, z + h],  # 5: 右下后
                    [x + l, y + w, z + h], # 6: 右上后
                    [x, y + w, z + h]   # 7: 左上后
                ])
                
                # 定义6个面的顶点索引
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
                    [vertices[0], vertices[1], vertices[5], vertices[4]],    # 前面
                    [vertices[2], vertices[3], vertices[7], vertices[6]],    # 后面
                    [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
                    [vertices[1], vertices[2], vertices[6], vertices[5]]    # 右面
                ]
                
                # 创建3D多边形集合
                face_collection = Poly3DCollection(faces, alpha=0.7, facecolor=colors[idx], 
                                                 edgecolor='black', linewidths=0.5)
                ax.add_collection3d(face_collection)
            
            # 绘制容器边框（透明）
            container_vertices = np.array([
                [0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0],
                [0, 0, H], [L, 0, H], [L, W, H], [0, W, H]
            ])
            
            # 绘制容器边框线
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
            ]
            
            for edge in edges:
                points = container_vertices[edge]
                ax.plot3D(*points.T, color='gray', linewidth=1, linestyle='--', alpha=0.3)
            
            # 设置坐标轴
            ax.set_xlabel('X (Length)', fontsize=12)
            ax.set_ylabel('Y (Width)', fontsize=12)
            ax.set_zlabel('Z (Height)', fontsize=12)
            
            # 设置坐标轴范围
            ax.set_xlim(0, L)
            ax.set_ylim(0, W)
            ax.set_zlim(0, H)
            
            # 设置标题
            utilization = container.volume_used / container.volume_capacity
            full_title = f"{title}\nUtilization: {utilization:.2%} | Items: {num_items}"
            ax.set_title(full_title, fontsize=14, fontweight='bold', pad=20)
            
            # 设置视角
            ax.view_init(elev=20, azim=45)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 3D可视化已保存到: {save_path}")
            
            # 显示图片
            plt.show()
    
        except Exception as e:
            print(f"⚠️ 3D可视化失败: {e}")
            return

    def _load_model(self):
        """加载训练好的模型"""
        try:
            state_dict = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            print(f"✅ 成功加载模型权重")
        except FileNotFoundError:
            print(f"❌ 错误: 找不到模型文件 {self.model_path}")
            raise
        except Exception as e:
            print(f"❌ 加载模型失败: {str(e)}")
            raise
    
    def _calculate_volume(self, item_data):
        """计算货物体积"""
        if len(item_data) == 4:
            _, length, width, height = item_data
        elif len(item_data) == 3:
            length, width, height = item_data
        else:
            raise ValueError(f"物品数据格式错误，应为3或4个元素: {item_data}")
        return length * width * height
    
    def _prepare_tensors(self, state):
        """将状态转换为张量"""
        if state is None:
            return None
        
        tensor_state = {
            'occupancy_multiscale': {
                'quarter': torch.FloatTensor(state['occupancy_multiscale']['quarter']).unsqueeze(0)
            },
            'candidates': torch.FloatTensor(state['candidates']).unsqueeze(0),
            'current_item': torch.FloatTensor(state['current_item']).unsqueeze(0),
            'remaining_items': torch.FloatTensor(state['remaining_items']).unsqueeze(0),
            'global_features': torch.FloatTensor(state['global_features']).unsqueeze(0),
            'height_map': torch.FloatTensor(state['height_map']).unsqueeze(0),
            'lookahead_items': torch.FloatTensor(state['lookahead_items']).unsqueeze(0)
        }
        return tensor_state
    
    def _select_best_action(self, state_tensor, valid_actions):
        """用 Actor 对所有有效动作打分，返回得分最高的动作（贪心）。"""
        if not valid_actions:
            return None

        L, W, H = self.container_size
        feats = []
        for pos, rot in valid_actions:
            x, y, z = pos
            feats.append([x / max(L - 1, 1),
                          y / max(W - 1, 1),
                          z / max(H - 1, 1),
                          rot / 5.0])
        action_feats = torch.FloatTensor(feats)  # [N, 4]

        with torch.no_grad():
            fused = self.model.encode_state(state_tensor)        # [1, 128]
            scores = self.model.score_actions(fused, action_feats)  # [N]

        best_idx = scores.argmax().item()
        return valid_actions[best_idx]
    
    def test_single_item(self, item_data, container_state=None):
        """
        测试单个物品的放置（保持兼容性）
        参数:
        item_data: (长, 宽, 高) 或 (数据编号, 长, 宽, 高)
        container_state: 容器状态（可选，用于继续之前的测试）
        
        返回:
        result: 放置结果字典
        """
        # 处理输入数据格式
        if len(item_data) == 4:
            data_id, length, width, height = item_data
        elif len(item_data) == 3:
            data_id = None
            length, width, height = item_data
        else:
            raise ValueError(f"物品数据格式错误，应为3或4个元素: {item_data}")
        
        volume = length * width * height
        
        # 创建环境或使用现有状态
        if container_state is None:
            env = SupervisedBinPackingEnv(self.container_size)
            env.reset()
        else:
            env = container_state
        
        # 添加物品到环境
        state = env.add_item((length, width, height))
        
        if state is None:
            return {
                'success': False,
                'message': '无法添加物品到环境',
                'data_id': data_id,
                'item_size': (length, width, height),
                'volume': volume
            }
        
        # 获取有效动作
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return {
                'success': False,
                'message': '没有有效的放置位置',
                'data_id': data_id,
                'item_size': (length, width, height),
                'volume': volume,
                'container_utilization': env.container.volume_used / env.container.volume_capacity
            }
        
        # 准备状态张量
        state_tensor = self._prepare_tensors(state)
        if state_tensor is None:
            return {
                'success': False,
                'message': '状态张量准备失败',
                'data_id': data_id,
                'item_size': (length, width, height),
                'volume': volume
            }
        
        # 选择最佳动作
        best_action = self._select_best_action(state_tensor, valid_actions)
        
        if best_action is None:
            return {
                'success': False,
                'message': '无法选择有效动作',
                'data_id': data_id,
                'item_size': (length, width, height),
                'volume': volume
            }
        
        # 执行动作
        position, rotation = best_action
        next_state, reward, done = env.step(best_action)
        
        success = (env.current_item is None)  # 成功放置后current_item会被清空
        
        result = {
            'success': success,
            'data_id': data_id,
            'item_size': (length, width, height),
            'volume': volume,
            'position': position,
            'rotation': rotation,
            'reward': reward,
            'container_utilization': env.container.volume_used / env.container.volume_capacity,
            'env': env  # 返回环境对象以便继续使用
        }
        
        if not success:
            result['message'] = '放置失败，可能因为位置冲突'
        
        return result
    
    def test_multiple_items_with_lookahead(self, items_data):
        """
        使用预见窗口排序策略测试多个物品的连续放置
        参数:
        items_data: 物品数据列表，每个元素为 (数据编号, 长, 宽, 高) 或 (长, 宽, 高)
        
        返回:
        results: 所有物品的放置结果列表
        """
        results = []
        env = None
        placed_count = 0
        
        # 创建物品队列
        items_queue = list(items_data)
        window = []  # 预见窗口
        
        print(f"\n{'='*60}")
        print(f"开始预见窗口测试 {len(items_data)} 个物品")
        print(f"预见窗口大小: k={self.k}")
        print(f"{'='*60}\n")
        
        step = 0
        
        while items_queue or window:
            step += 1
            
            # 填充预见窗口到k个物品
            while len(window) < self.k and items_queue:
                window.append(items_queue.pop(0))
            
            if not window:
                break
            
            # 按体积从大到小排序窗口中的物品
            window_with_volume = []
            for item in window:
                volume = self._calculate_volume(item)
                window_with_volume.append((item, volume))
            
            # 排序：体积大的优先
            window_with_volume.sort(key=lambda x: x[1], reverse=True)
            
            # 选择体积最大的物品进行装载
            selected_item, selected_volume = window_with_volume[0]
            
            # 从窗口中移除选中的物品
            window.remove(selected_item)
            
            # 测试选中的物品
            result = self.test_single_item(selected_item, env)
            results.append(result)
            
            # 添加窗口信息到结果
            window_info = {
                'step': step,
                'window_size': len(window) + 1,  # +1 因为刚移除了一个
                'selected_volume': selected_volume,
                'window_volumes': [v for _, v in window_with_volume[1:]]  # 窗口中剩余物品的体积
            }
            result['window_info'] = window_info
            
            # 更新环境状态以便继续
            if result['success'] and 'env' in result:
                env = result['env']
                env.current_item = None  # 确保清空当前物品
                placed_count += 1
                
                print(f"✅ 步骤 {step}: 物品 [ID: {result['data_id']}] "
                      f"尺寸: {result['item_size']} (体积: {selected_volume}) → "
                      f"位置: {result['position']} 利用率: {result['container_utilization']:.2%}")
                print(f"   窗口状态: 当前窗口大小={len(window)}, 剩余队列={len(items_queue)}")
                
            else:
                print(f"❌ 步骤 {step}: 物品 [ID: {result.get('data_id', 'N/A')}] "
                      f"尺寸: {result['item_size']} (体积: {selected_volume}) → "
                      f"放置失败: {result.get('message', '未知错误')}")
                print(f"   窗口状态: 当前窗口大小={len(window)}, 剩余队列={len(items_queue)}")
            
            # 如果容器已满，停止
            if env and env.container.volume_used >= env.container.volume_capacity * 0.99:
                print(f"\n⚠️ 容器已满，停止测试剩余物品")
                # 将窗口和队列中剩余物品标记为未处理
                remaining_items = window + items_queue
                for item in remaining_items:
                    volume = self._calculate_volume(item)
                    data_id = item[0] if len(item) == 4 else None
                    item_size = item[1:] if len(item) == 4 else item
                    
                    result = {
                        'success': False,
                        'message': '容器已满，无法继续放置',
                        'data_id': data_id,
                        'item_size': item_size,
                        'volume': volume,
                        'container_utilization': env.container.volume_used / env.container.volume_capacity
                    }
                    results.append(result)
                break
        
        # 打印汇总信息
        print(f"\n{'='*60}")
        print(f"预见窗口测试完成汇总")
        print(f"{'='*60}")
        print(f"预见窗口大小: k={self.k}")
        print(f"总物品数: {len(items_data)}")
        print(f"成功放置: {placed_count}")
        print(f"失败数量: {len(items_data) - placed_count}")
        if env:
            final_utilization = env.container.volume_used / env.container.volume_capacity
            print(f"最终利用率: {final_utilization:.2%}")
        
        # 计算平均处理体积
        placed_results = [r for r in results if r.get('success', False)]
        if placed_results:
            avg_volume = np.mean([r['volume'] for r in placed_results])
            print(f"平均放置物品体积: {avg_volume:.2f}")
        
        print(f"{'='*60}\n")
        
        # 不在此处可视化；可视化将在所有 episode 完成后统一处理
        
        return results
    
    def test_multiple_items(self, items_data):
        """兼容原有接口的方法"""
        return self.test_multiple_items_with_lookahead(items_data)


def test_from_list_with_lookahead(items_data, k=None, model_path=None):
    """
    便捷函数：使用预见窗口从列表测试物品
    参数:
    items_data: 物品数据列表，格式为 [(数据编号, 长, 宽, 高), ...] 或 [(长, 宽, 高), ...]
    k: 预见窗口大小
    model_path: 模型路径（可选）
    
    示例:
    items = [
        (1, 3, 3, 3),
        (2, 2, 2, 2),
        (3, 4, 4, 4),
        (4, 1, 5, 3)
    ]
    results = test_from_list_with_lookahead(items, k=2)
    """
    tester = ModelTesterWithLookahead(model_path, k)
    return tester.test_multiple_items_with_lookahead(items_data)


def test_from_excel_with_lookahead(excel_file=None, data_scale=None, episode_id=None, k=None, model_path=None):
    """
    使用预见窗口从Excel文件读取数据并测试
    参数:
    excel_file: Excel文件路径，如果为None则使用测试默认文件（数据_101010_cut2.xlsx）
    data_scale: 数据规模，如果为None则使用parameter.py中的默认值
    episode_id: episode编号，如果为None则测试所有episode
    k: 预见窗口大小，如果为None则使用parameter.py中的LOOKAHEAD_K
    model_path: 模型路径（可选）
    
    返回:
    results: 测试结果
    
    示例:
    # 测试特定episode，使用k=2预见窗口
    results = test_from_excel_with_lookahead(episode_id=1, k=2)
    
    # 测试所有episode，使用默认预见窗口
    results = test_from_excel_with_lookahead()
    """
    # 使用默认参数
    if excel_file is None:
        excel_file = '数据_101010_cut1.xlsx'  # 测试文件
    if data_scale is None:
        try:
            import pandas as pd
            import re
            xl = pd.ExcelFile(excel_file)
            candidates = [s for s in xl.sheet_names if s.startswith('规模')]
            def parse_scale(name):
                m = re.search(r'\d+', name)
                return int(m.group()) if m else None
            parsed = [(s, parse_scale(s)) for s in candidates]
            parsed = [t for t in parsed if t[1] is not None]
            if parsed:
                # 若文件中存在与 DATA_SCALE 相同的规模则优先，否则取第一个匹配
                preferred = next((t for t in parsed if t[1] == DATA_SCALE), None)
                data_scale = preferred[1] if preferred else parsed[0][1]
        except Exception:
            pass
        if data_scale is None:
            data_scale = DATA_SCALE
    
    if k is None:
        k = LOOKAHEAD_K
    
    print(f"\n{'='*60}")
    print(f"从Excel文件读取测试数据 - 预见窗口版本")
    print(f"{'='*60}")
    print(f"Excel文件: {excel_file}")
    print(f"数据规模: {data_scale}")
    print(f"预见窗口大小: k={k}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 加载Excel数据
    try:
        data = load_excel_data(excel_file, data_scale)
    except Exception as e:
        print(f"❌ 读取Excel文件失败: {str(e)}")
        return None
    
    # 创建测试器
    tester = ModelTesterWithLookahead(model_path, k)
    
    # 如果指定了episode_id，只测试该episode
    if episode_id is not None:
        print(f"\n测试Episode {episode_id}")
        items_data, optimal_positions = extract_episode_data_corrected(data, episode_id)
        
        if not items_data:
            print(f"❌ Episode {episode_id} 没有数据")
            return None
        
        # 将items_data转换为测试格式（包含数据编号）
        test_items = []
        episode_data = data[data[:, 0] == episode_id]
        for row in episode_data:
            data_id = int(row[0])
            length = int(row[1])
            width = int(row[2])
            height = int(row[3])
            test_items.append((data_id, length, width, height))
        
            results = tester.test_multiple_items_with_lookahead(test_items)
            
            # 获取最终容器状态（不在此处可视化）
            final_env = None
            final_utilization = None
            if results and len(results) > 0:
                last_result = results[-1]
                if 'env' in last_result and last_result['env']:
                    final_env = last_result['env']
                    try:
                        final_utilization = final_env.container.volume_used / final_env.container.volume_capacity
                    except Exception:
                        final_utilization = None

            return {
                'episode_id': episode_id,
                'results': results,
                'optimal_positions': optimal_positions,
                'lookahead_k': k,
                'final_env': final_env,
                'final_utilization': final_utilization
            }
    
    # 测试所有episode
    else:
        unique_episodes = np.unique(data[:, 0])
        print(f"\n发现 {len(unique_episodes)} 个不同的episode")
        
        all_results = {}
        
        for ep_id in unique_episodes:
            print(f"\n{'='*60}")
            print(f"测试Episode {ep_id} - 预见窗口 k={k}")
            print(f"{'='*60}")
            
            items_data, optimal_positions = extract_episode_data_corrected(data, ep_id)
            
            if not items_data:
                print(f"⚠️ Episode {ep_id} 没有数据，跳过")
                continue
            
            # 将items_data转换为测试格式（包含数据编号）
            test_items = []
            episode_data = data[data[:, 0] == ep_id]
            for row in episode_data:
                data_id = int(row[0])
                length = int(row[1])
                width = int(row[2])
                height = int(row[3])
                test_items.append((data_id, length, width, height))
            
            results = tester.test_multiple_items_with_lookahead(test_items)
            
            # 获取最终容器状态（在所有episode结束后可视化最高利用率的episode）
            final_env = None
            final_utilization = None
            if results and len(results) > 0:
                last_result = results[-1]
                if 'env' in last_result and last_result['env']:
                    final_env = last_result['env']
                    try:
                        final_utilization = final_env.container.volume_used / final_env.container.volume_capacity
                    except Exception:
                        final_utilization = None
            
            all_results[ep_id] = {
                'results': results,
                'optimal_positions': optimal_positions,
                'lookahead_k': k,
                'final_env': final_env,
                'final_utilization': final_utilization
            }
        
        # 打印汇总信息
        print(f"\n{'='*60}")
        print(f"所有Episode预见窗口测试完成 (k={k})")
        print(f"{'='*60}")
        total_episodes = len(all_results)
        print(f"测试了 {total_episodes} 个episode")
        
        # 计算平均利用率和平均处理体积
        total_utilization = 0
        total_items = 0
        placed_items = 0
        total_volume = 0
        for ep_id, ep_result in all_results.items():
            ep_results = ep_result['results']
            if ep_results:
                last_result = ep_results[-1]
                if 'container_utilization' in last_result:
                    total_utilization += last_result['container_utilization']
                for r in ep_results:
                    if r.get('success', False):
                        placed_items += 1
                        total_volume += r.get('volume', 0)
                    total_items += 1
        
        if total_episodes > 0:
            avg_utilization = total_utilization / total_episodes
            placement_rate = placed_items / total_items if total_items > 0 else 0
            avg_volume = total_volume / placed_items if placed_items > 0 else 0
            print(f"平均利用率: {avg_utilization:.2%}")
            print(f"物品放置率: {placement_rate:.2%} ({placed_items}/{total_items})")
            print(f"平均处理物品体积: {avg_volume:.2f}")
            # 计算并打印平均每个箱子装载多少个货物
            try:
                avg_items_per_box = placed_items / total_episodes if total_episodes > 0 else 0
            except Exception:
                avg_items_per_box = 0
            print(f"平均每个箱子装载货物数量: {avg_items_per_box:.2f} ({placed_items}/{total_episodes})")
            # 打印总耗时
            try:
                elapsed_seconds = time.time() - start_time
                m, s = divmod(int(elapsed_seconds), 60)
                h, m = divmod(m, 60)
                print(f"总耗时: {h:d}h {m:d}m {s:d}s ({elapsed_seconds:.2f}s)")
            except Exception:
                pass
        
        print(f"{'='*60}\n")
        # 找出最终利用率最高的 episode 并可视化其装载图
        best_ep = None
        best_util = -1.0
        for ep_id, ep_info in all_results.items():
            util = ep_info.get('final_utilization')
            if util is not None and util > best_util:
                best_util = util
                best_ep = ep_id

        if best_ep is not None:
            best_env = all_results[best_ep].get('final_env')
            if best_env and getattr(best_env, 'container', None) and best_env.container.items:
                print(f"\n🎨 生成最高利用率的 Episode {best_ep} (利用率: {best_util:.2%}) 的3D可视化图...")
                tester._visualize_packing_3d(
                    best_env.container,
                    title=f"Best Episode {best_ep} - Utilization {best_util:.2%}",
                    save_path=f"packing_best_episode_{best_ep}.png"
                )

        return all_results


def batch_test_k_datasets(k_min=1, k_max=8, datasets=None, save_dir=r"C:\Users\Administrator\Desktop\A3C1\A3C\图", out_excel_name="batch_summary.xlsx"):
    """
    批量测试不同 k 值和多个数据集，保存每次测试的汇总到 Excel，并把每个数据集在该 k 下的最佳装载图保存为 tif。
    保存路径默认: save_dir，Excel 名称为 out_excel_name。
    """
    if datasets is None:
        datasets = ['数据_101010_cut1.xlsx', '数据_101010_cut2.xlsx', '数据_101010_rs.xlsx']

    os.makedirs(save_dir, exist_ok=True)
    rows = []

    for k_val in range(k_min, k_max + 1):
        for excel_file in datasets:
            print(f"\n--- 开始: k={k_val} 数据集={excel_file} ---")
            t0 = time.time()
            result = test_from_excel_with_lookahead(excel_file=excel_file, k=k_val, model_path=None)
            elapsed = time.time() - t0

            # 解析返回结果（可能为单 episode 的 dict，也可能为所有 episode 的 mapping）
            total_episodes = 0
            avg_utilization = None
            placement_rate = 0.0
            avg_volume = 0.0
            avg_items_per_box = 0.0
            total_items = 0
            placed_items = 0

            best_env = None
            best_util = -1.0

            if result is None:
                print(f"⚠️ 未获得结果: {excel_file} k={k_val}")
                continue

            # single-episode返回（包含 'episode_id'）
            if isinstance(result, dict) and 'episode_id' in result:
                total_episodes = 1
                ep = result
                res_list = ep.get('results', []) or []
                for r in res_list:
                    total_items += 1
                    if r.get('success', False):
                        placed_items += 1
                        avg_volume += r.get('volume', 0)
                avg_volume = (avg_volume / placed_items) if placed_items > 0 else 0
                placement_rate = placed_items / total_items if total_items > 0 else 0
                avg_items_per_box = placed_items / total_episodes if total_episodes > 0 else 0
                avg_utilization = ep.get('final_utilization')
                best_env = ep.get('final_env')
                best_util = avg_utilization or -1.0
            else:
                # 假定为 mapping of episodes
                all_results = result
                total_episodes = len(all_results)
                total_util = 0.0
                util_count = 0
                total_volume = 0.0
                for ep_id, ep in all_results.items():
                    if ep is None:
                        continue
                    fu = ep.get('final_utilization')
                    if fu is not None:
                        total_util += fu
                        util_count += 1
                    ep_results = ep.get('results', []) or []
                    for r in ep_results:
                        total_items += 1
                        if r.get('success', False):
                            placed_items += 1
                            total_volume += r.get('volume', 0)
                    # 寻找最佳 episode
                    if fu is not None and fu > best_util:
                        best_util = fu
                        best_env = ep.get('final_env')

                avg_utilization = (total_util / util_count) if util_count > 0 else None
                placement_rate = placed_items / total_items if total_items > 0 else 0
                avg_volume = (total_volume / placed_items) if placed_items > 0 else 0
                avg_items_per_box = placed_items / total_episodes if total_episodes > 0 else 0

            # 保存最佳装载图（若存在）
            dataset_name = os.path.splitext(os.path.basename(excel_file))[0]
            img_name = f"k=({k_val}){dataset_name}.tif"
            img_path = os.path.join(save_dir, img_name)
            if best_env and getattr(best_env, 'container', None) and best_env.container.items:
                # 使用一个临时 tester 来绘图（无需加载模型）
                tmp_tester = ModelTesterWithLookahead(model_path=None, k=k_val)
                try:
                    tmp_tester._visualize_packing_3d(
                        best_env.container,
                        title=f"{dataset_name} k={k_val} Util {best_util:.2%}" if best_util >= 0 else f"{dataset_name} k={k_val}",
                        save_path=img_path
                    )
                except Exception as e:
                    print(f"⚠️ 保存图像失败: {e}")

            # 格式化耗时字符串
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            time_str = f"{h:d}h {m:d}m {s:d}s"

            # 格式化为与脚本控制台输出相同的信息（便于在 Excel 中查看）
            avg_util_pct = f"{avg_utilization:.2%}" if avg_utilization is not None else ''
            placement_rate_pct = f"{placement_rate:.2%}"
            placement_rate_str = f"{placement_rate_pct} ({placed_items}/{total_items})"
            avg_items_per_box_str = f"{avg_items_per_box:.2f} ({placed_items}/{total_episodes})" if total_episodes > 0 else f"{avg_items_per_box:.2f}"

            rows.append({
                'k': k_val,
                'dataset': dataset_name,
                'total_episodes': total_episodes,
                'avg_utilization': avg_utilization if avg_utilization is not None else '',
                'avg_utilization_str': avg_util_pct,
                'placement_rate': placement_rate,
                'placement_rate_str': placement_rate_str,
                'avg_volume': avg_volume,
                'avg_items_per_box': avg_items_per_box,
                'avg_items_per_box_str': avg_items_per_box_str,
                'total_items': total_items,
                'placed_items': placed_items,
                'elapsed_seconds': elapsed,
                'elapsed_str': time_str
            })

    # 保存 Excel
    if rows:
        df = pd.DataFrame(rows)
        out_path = os.path.join(save_dir, out_excel_name)
        try:
            df.to_excel(out_path, index=False)
            print(f"\n✅ 批量测试结果已保存到: {out_path}")
        except Exception as e:
            print(f"❌ 保存Excel失败: {e}")
    else:
        print("⚠️ 未生成任何结果行，未保存 Excel。")

    return rows


# 保持向后兼容的接口
def test_from_list(items_data, model_path=None):
    """兼容原有接口"""
    return test_from_list_with_lookahead(items_data, None, model_path)


def test_from_array(items_array):
    """兼容原有接口"""
    items_data = [tuple(row) for row in items_array]
    return test_from_list_with_lookahead(items_data)


def test_from_excel(excel_file=None, data_scale=None, episode_id=None, model_path=None):
    """兼容原有接口"""
    return test_from_excel_with_lookahead(excel_file, data_scale, episode_id, None, model_path)


# 原有的ModelTester类别名，保持兼容性
ModelTester = ModelTesterWithLookahead


if __name__ == '__main__':
    print("="*60)
    print("模型测试脚本 - 预见窗口排序版本")
    print("="*60)
    
    import sys
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试3D装箱模型 - 支持预见窗口')
    parser.add_argument('episode_id', nargs='?', type=int, help='Episode ID (可选)')
    parser.add_argument('--k', type=int, default=LOOKAHEAD_K, help=f'预见窗口大小 (默认: {LOOKAHEAD_K})')
    # 批量测试选项
    parser.add_argument('--batch', action='store_true', help='批量运行 k 值与多个数据集的测试 (调用 batch_test_k_datasets)')
    parser.add_argument('--k-min', type=int, default=1, help='批量测试的最小 k 值 (默认: 1)')
    parser.add_argument('--k-max', type=int, default=8, help='批量测试的最大 k 值 (默认: 8)')
    parser.add_argument('--datasets', nargs='+', default=None, help='要批量测试的 Excel 数据集列表，空格分隔（例如: 数据_101010_cut1.xlsx 数据_101010_cut2.xlsx 数据_101010_rs.xlsx）')
    parser.add_argument('--out-dir', type=str, default=r"C:\Users\Administrator\Desktop\A3C1\A3C\图", help='批量测试输出目录 (默认: C:\\Users\\Administrator\\Desktop\\A3C1\\A3C\\图)')
    parser.add_argument('--out-excel', type=str, default='batch_summary.xlsx', help='批量测试输出的 Excel 文件名 (默认: batch_summary.xlsx)')
    
    args = parser.parse_args()
    
    # 如果没有传入任何命令行参数（仅脚本名），默认执行批量测试
    if len(sys.argv) == 1:
        print("未提供命令行参数，默认执行批量测试 (k=2..8, 默认数据集)")
        batch_test_k_datasets()
        sys.exit(0)

    if args.batch:
        print(f"启动批量测试: k_range={args.k_min}..{args.k_max}, datasets={args.datasets or '默认列表'}, out_dir={args.out_dir}")
        batch_test_k_datasets(k_min=args.k_min, k_max=args.k_max, datasets=args.datasets, save_dir=args.out_dir, out_excel_name=args.out_excel)
        sys.exit(0)

    if args.episode_id is not None:
        print(f"从Excel文件读取数据，测试Episode {args.episode_id}，预见窗口 k={args.k}")
        results = test_from_excel_with_lookahead(episode_id=args.episode_id, k=args.k)
    else:
        print(f"从Excel文件读取数据，测试所有episode，预见窗口 k={args.k}")
        print("提示: 可以使用 'python test_model.py <episode_id> --k <窗口大小>' 测试特定episode")
        
        # 测试所有episode
        results = test_from_excel_with_lookahead(k=args.k)
    
    if results is None:
        print("❌ 测试失败或没有数据")
    else:
        print("✅ 预见窗口测试完成")