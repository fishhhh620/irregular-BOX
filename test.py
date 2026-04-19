"""
测试相关函数
"""
from my_imports import *
from parameter_irregular import EXCEL_FILE, DATA_SCALE
from data_loader import load_excel_data, extract_episode_data_corrected, OnlineItemIterator


def test_excel_data_loading_corrected():
    """修正版：测试Excel数据加载功能"""
    print("测试Excel数据加载...")

    try:
        # 测试数据加载
        data = load_excel_data(EXCEL_FILE, DATA_SCALE)
        print(f"数据加载成功，形状: {data.shape}")

        # 显示前几行数据
        print("前5行数据:")
        print(data[:5])

        # 测试数据分组
        unique_episodes = np.unique(data[:, 0])
        print(f"发现 {len(unique_episodes)} 个episode: {unique_episodes[:10]}...")

        # 测试提取单个episode数据（使用修正版函数）
        episode_id = unique_episodes[0]
        items_data, positions_data, action_targets = extract_episode_data_corrected(data, episode_id)
        print(f"Episode {episode_id} 包含 {len(items_data)} 个物品")
        print(f"前3个物品: {items_data[:3]}")
        print(f"前3个最优位置: {positions_data[:3]}")
        print(f"前3个动作目标(货物编号,空间编号,空间最大值): {action_targets[:3]}")

        return True

    except Exception as e:
        print(f"数据加载测试失败: {str(e)}")
        return False


def test_online_mode():
    """测试Online模式：逐个读取货物"""
    print("\n测试Online模式...")
    
    try:
        # 加载数据
        data = load_excel_data(EXCEL_FILE, DATA_SCALE)
        unique_episodes = np.unique(data[:, 0])
        episode_id = unique_episodes[0]
        
        # 提取episode数据
        items_data, positions_data, action_targets = extract_episode_data_corrected(data, episode_id)

        if not items_data:
            print("没有找到测试数据")
            return False

        # 创建Online迭代器
        iterator = OnlineItemIterator(items_data, positions_data, action_targets)

        print(f"开始逐个读取货物（共 {iterator.total_items} 个）...")

        count = 0
        while iterator.has_next():
            result = iterator.get_next_item()
            if result:
                item_data, optimal_pos, action_target = result
                count += 1
                print(f"  货物 {count}: 尺寸={item_data}, 位置={optimal_pos}, 动作={action_target}")
                
                # 只显示前3个
                if count >= 3:
                    print(f"  ... (还有 {iterator.total_items - count} 个货物)")
                    break
        
        print(f"✅ Online模式测试通过：成功读取 {count} 个货物（每次只读取1个）")
        return True
        
    except Exception as e:
        print(f"❌ Online模式测试失败: {str(e)}")
        return False


if __name__ == '__main__':
    print("测试Excel数据加载...")
    print("=" * 60)

    if test_excel_data_loading_corrected():
        print("\n数据加载测试通过！")
        
        # 测试Online模式
        test_online_mode()
    else:
        print("\n数据加载测试失败，请检查Excel文件和配置")

