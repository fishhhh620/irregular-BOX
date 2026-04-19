"""
Worker：按 episode 粒度区分监督 / 纯 RL 训练。

  有标签 episode（episode_id in labeled_episode_ids）：
      损失 = NLL Loss（向专家位置靠拢）+ VALUE_LOSS_COEF * Critic Loss
  无标签 episode（纯 RL）：
      损失 = Actor Loss + VALUE_LOSS_COEF * Critic Loss - ENTROPY_COEF * Entropy
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from my_imports import *
from parameter_irregular import *
from network_irregular import IrregularA3CNet as ImprovedA3CNet
from environment_irregular import IrregularBinPackingEnv as SupervisedBinPackingEnv
from data_loader import extract_episode_data_corrected, OnlineItemIterator
from model_saver import ModelSaver


class SupervisedWorkerAgent(mp.Process):

    def __init__(self, global_net, optimizer, global_ep_idx, global_ep_r, res_queue,
                 name, episode_ids, excel_data, model_saver=None, labeled_episode_ids=None):
        super().__init__()
        self.name = f'worker_{name:02d}'
        self.global_net = global_net
        self.optimizer = optimizer
        self.global_ep_idx = global_ep_idx
        self.global_ep_r = global_ep_r
        self.res_queue = res_queue
        self.model_saver = model_saver
        self.episode_ids = episode_ids
        self.excel_data = excel_data
        self.labeled_episode_ids = labeled_episode_ids or set()
        self.local_net = ImprovedA3CNet(CONTAINER_SIZE)
        print(f"{self.name} 分配了 {len(episode_ids)} 个episodes，"
              f"其中有标签 {sum(1 for e in episode_ids if e in self.labeled_episode_ids)} 个")

    # ------------------------------------------------------------------ #
    #  辅助方法                                                             #
    # ------------------------------------------------------------------ #

    def _prepare_tensors(self, state):
        if state is None:
            return None
        return {
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

    def _encode_actions(self, valid_actions):
        """有效动作 → 归一化特征张量 [N, 4]"""
        L, W, H = CONTAINER_SIZE
        feats = []
        for pos, rot in valid_actions:
            x, y, z = pos
            feats.append([x / max(L - 1, 1),
                          y / max(W - 1, 1),
                          z / max(H - 1, 1),
                          rot / 5.0])
        return torch.FloatTensor(feats)

    def _find_expert_action_idx(self, valid_actions, expert_pos):
        """在 valid_actions 中找到与专家位置（x,y,z）最近的动作索引。"""
        if not valid_actions or expert_pos is None:
            return -1
        best_idx, min_dist = 0, float('inf')
        for i, (pos, _) in enumerate(valid_actions):
            dist = sum((a - b) ** 2 for a, b in zip(pos, expert_pos))
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return best_idx

    def _select_action(self, state_tensor, valid_actions):
        """网络对所有有效动作打分，采样返回 (action, log_prob, entropy)。"""
        self.local_net.eval()
        fused        = self.local_net.encode_state(state_tensor)
        action_feats = self._encode_actions(valid_actions)
        scores       = self.local_net.score_actions(fused, action_feats)
        probs        = F.softmax(scores, dim=0)
        m            = Categorical(probs)
        action_idx   = m.sample().item()
        return valid_actions[action_idx], m.log_prob(torch.tensor(action_idx)), m.entropy()

    # ------------------------------------------------------------------ #
    #  网络更新                                                             #
    # ------------------------------------------------------------------ #

    def _update_network(self, buffer_s, buffer_a, buffer_r, buffer_v, buffer_logp,
                        next_state, done, buffer_entropy,
                        is_supervised=False,
                        buffer_valid_actions=None, buffer_expert_idx=None):

        # ---- 折扣回报 ----
        if done:
            v_s_ = 0.0
        else:
            ns = self._prepare_tensors(next_state)
            if ns is not None:
                self.local_net.eval()
                with torch.no_grad():
                    _, v_s_ = self.local_net(ns)
                v_s_ = v_s_.item()
            else:
                v_s_ = 0.0

        v_targets = []
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            v_targets.append(v_s_)
        v_targets.reverse()
        v_target_t = torch.FloatTensor(v_targets)
        v_t        = torch.FloatTensor(buffer_v)

        # ---- 批量 Critic ----
        batch = {}
        for key in ['candidates', 'current_item', 'remaining_items', 'global_features', 'height_map']:
            batch[key] = torch.FloatTensor(np.stack([s[key] for s in buffer_s]))
        batch['occupancy_multiscale'] = {
            'quarter': torch.FloatTensor(np.stack([s['occupancy_multiscale']['quarter'] for s in buffer_s]))
        }
        batch['lookahead_items'] = torch.FloatTensor(np.stack([s['lookahead_items'] for s in buffer_s]))

        self.local_net.train()
        _, values = self.local_net(batch)
        values = values.squeeze()
        critic_loss = F.mse_loss(values, v_target_t)

        # ---- 有标签 episode：NLL Loss ----
        if is_supervised and buffer_valid_actions and buffer_expert_idx:
            nll_log_probs = []
            for i in range(len(buffer_s)):
                if buffer_expert_idx[i] < 0:
                    continue
                s_t = self._prepare_tensors(buffer_s[i])
                if s_t is None:
                    continue
                feats  = self._encode_actions(buffer_valid_actions[i])
                self.local_net.train()
                fused  = self.local_net.encode_state(s_t)
                scores = self.local_net.score_actions(fused, feats)
                log_p  = F.log_softmax(scores, dim=0)[buffer_expert_idx[i]]
                nll_log_probs.append(log_p)

            nll_loss   = (-torch.stack(nll_log_probs).mean()
                          if nll_log_probs else torch.tensor(0.0))
            total_loss = nll_loss + VALUE_LOSS_COEF * critic_loss

        # ---- 无标签 episode：标准 A3C Loss ----
        else:
            advantage    = v_target_t - v_t
            logp_t       = torch.stack(buffer_logp)
            actor_loss   = -(logp_t * advantage.detach()).mean()
            entropy_mean = (torch.stack(buffer_entropy).mean()
                            if buffer_entropy else torch.tensor(0.0))
            total_loss   = (actor_loss
                            + VALUE_LOSS_COEF * critic_loss
                            - ENTROPY_COEF * entropy_mean)

        # ---- 梯度同步 ----
        self.optimizer.zero_grad()
        total_loss.backward()
        for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
            if lp.grad is not None:
                if gp.grad is None:
                    gp.grad = lp.grad.clone()
                else:
                    gp.grad += lp.grad
        nn.utils.clip_grad_norm_(self.global_net.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        self.local_net.load_state_dict(self.global_net.state_dict())

    # ------------------------------------------------------------------ #
    #  主训练循环                                                           #
    # ------------------------------------------------------------------ #

    def run(self):
        total_step    = 1
        episode_cycle = 0

        try:
            while self.global_ep_idx.value < MAX_EPISODES:
                episode_id   = self.episode_ids[episode_cycle % len(self.episode_ids)]
                is_supervised = episode_id in self.labeled_episode_ids

                # 有标签 episode 提供位置 + 动作目标；无标签只提供货物尺寸
                items_data, optimal_placements, action_targets = \
                    extract_episode_data_corrected(self.excel_data, episode_id)
                if not is_supervised:
                    optimal_placements = []
                    action_targets     = []

                if not items_data:
                    episode_cycle += 1
                    continue

                item_iterator = OnlineItemIterator(items_data, optimal_placements, action_targets)
                env = SupervisedBinPackingEnv(CONTAINER_SIZE, optimal_placements)
                env.reset()

                buf_s, buf_a, buf_r       = [], [], []
                buf_v, buf_logp, buf_ent  = [], [], []
                buf_va, buf_ei            = [], []   # valid_actions / expert_idx
                ep_r       = 0
                next_state = None

                while item_iterator.has_next():
                    result = item_iterator.get_next_item()
                    if result is None:
                        break
                    item_data, optimal_pos, _ = result

                    k   = max(1, min(LOOKAHEAD_K, K_MAX))
                    lah = item_iterator.peek_next_items(max(0, k - 1))
                    state = env.add_item(item_data, lah)
                    if state is None:
                        continue

                    valid_actions = env.get_valid_actions()
                    if not valid_actions:
                        ep_r += -10.0
                        continue

                    s_tensor = self._prepare_tensors(state)
                    if s_tensor is None:
                        continue

                    action, log_prob, entropy = self._select_action(s_tensor, valid_actions)

                    # 专家位置索引（无标签时为 -1）
                    expert_idx = (self._find_expert_action_idx(valid_actions, optimal_pos)
                                  if is_supervised else -1)

                    next_state, reward, _ = env.step(action)
                    ep_r += reward

                    buf_s.append(state);   buf_a.append(action);   buf_r.append(reward)
                    buf_logp.append(log_prob);  buf_ent.append(entropy)
                    buf_va.append(valid_actions);  buf_ei.append(expert_idx)

                    self.local_net.eval()
                    with torch.no_grad():
                        _, val = self.local_net(s_tensor)
                    buf_v.append(val.item())

                    if total_step % UPDATE_GLOBAL_ITER == 0 and buf_s:
                        self._update_network(buf_s, buf_a, buf_r, buf_v, buf_logp,
                                             next_state, False, buf_ent,
                                             is_supervised, buf_va, buf_ei)
                        buf_s, buf_a, buf_r     = [], [], []
                        buf_v, buf_logp, buf_ent = [], [], []
                        buf_va, buf_ei           = [], []
                    total_step += 1

                if buf_s:
                    self._update_network(buf_s, buf_a, buf_r, buf_v, buf_logp,
                                         None, True, buf_ent,
                                         is_supervised, buf_va, buf_ei)
                    buf_s, buf_a, buf_r     = [], [], []
                    buf_v, buf_logp, buf_ent = [], [], []
                    buf_va, buf_ei           = [], []

                ep_r += env.finish_episode()

                with self.global_ep_idx.get_lock():
                    self.global_ep_idx.value += 1
                    cur_ep = self.global_ep_idx.value

                with self.global_ep_r.get_lock():
                    if self.global_ep_r.value == 0:
                        self.global_ep_r.value = ep_r
                    else:
                        self.global_ep_r.value = self.global_ep_r.value * 0.99 + ep_r * 0.01

                util        = env.container.volume_used / env.container.volume_capacity
                placed      = env.items_placed
                total_items = item_iterator.total_items
                label_tag   = "S" if is_supervised else "R"

                self.res_queue.put((self.name, cur_ep, ep_r, util, placed, total_items))
                print(f'{self.name}[{label_tag}] Ep:{cur_ep} ID:{episode_id} '
                      f'r:{ep_r:.1f} util:{util:.2%} items:{placed}/{total_items}')

                if self.name == 'worker_00' and self.model_saver:
                    if cur_ep % 10 == 0:
                        self.model_saver.save_model(self.global_net.state_dict(), cur_ep)
                    if cur_ep % 50 == 0:
                        self.model_saver.save_backup(self.global_net.state_dict(), cur_ep)

                episode_cycle += 1

        finally:
            self.res_queue.put(None)
