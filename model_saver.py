"""
模型保存工具类
"""
import os
import torch

from my_imports import *


class ModelSaver:
    def __init__(self, save_path):
        self.save_path = save_path
        self.save_dir = os.path.dirname(save_path)
        self.ensure_directory()

    def ensure_directory(self):
        """确保保存目录存在"""
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"✅ 确保保存目录存在: {self.save_dir}")
        except Exception as e:
            print(f"❌ 创建保存目录失败: {str(e)}")

    def save_model(self, model_state_dict, episode_num):
        """保存模型"""
        try:
            torch.save(model_state_dict, self.save_path)
            print(f"✅ 模型已保存到: {self.save_path} (Episode: {episode_num})")
            return True
        except Exception as e:
            print(f"❌ 模型保存失败: {str(e)}")
            return False

    def save_backup(self, model_state_dict, episode_num):
        """保存备份模型"""
        backup_path = os.path.join(self.save_dir, f"model_backup_ep{episode_num}.pth")
        try:
            torch.save(model_state_dict, backup_path)
            print(f"💾 备份模型已保存到: {backup_path}")
            return True
        except Exception as e:
            print(f"⚠️ 备份保存失败: {str(e)}")
            return False

