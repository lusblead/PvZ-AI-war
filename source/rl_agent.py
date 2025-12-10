import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size):
        # 超参数设置
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # 经验回放池大小
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率初始值
        self.epsilon_min = 0.05  # 探索率最小值
        self.epsilon_decay = 0.995  # 探索率衰减率
        self.learning_rate = 0.001  # 学习率
        self.batch_size = 64  # 批次大小
        self.target_update_freq = 1000  # 目标网络更新频率
        self.gradient_clip = 1.0  # 梯度裁剪阈值

        # 构建策略网络和目标网络
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.train_step = 0

        # 奖励跟踪变量
        self.current_reward = 0
        self.achieved_time_intervals = set()
        self.zombie_enter_count = 0
        self.last_zombie_enter_time = 0

    def _build_model(self):
        """构建深度Q网络模型"""
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.state_size),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipvalue=self.gradient_clip)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def update_target_model(self):
        """更新目标网络权重"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放池"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """根据当前状态选择动作（ε-贪心策略）"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        """从经验回放池中采样并训练网络"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([t[0][0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3][0] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # 计算目标Q值
        target = rewards + self.gamma * np.amax(self.target_model.predict(next_states, verbose=0), axis=1) * (1 - dones)
        target_f = self.model.predict(states, verbose=0)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]

        # 训练模型
        self.model.fit(states, target_f, epochs=1, verbose=0)

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()

    def calculate_time_reward(self, remaining_time):
        """计算时间分段奖励"""
        time_passed = 300 - remaining_time
        reward = 0

        # 定义时间区间和对应奖励
        intervals = [
            (50, 1),  # 首次达到50秒
            (100, 3),  # 首次达到100秒
            (150, 5),  # 首次达到150秒
            (200, 8),  # 首次达到200秒
            (250, 12),  # 首次达到250秒
            (300, 20)  # 首次达到300秒
        ]

        for interval, reward_val in intervals:
            if time_passed >= interval and interval not in self.achieved_time_intervals:
                reward += reward_val
                self.achieved_time_intervals.add(interval)
                self.current_reward += reward_val

        return reward

    def calculate_zombie_punish(self, current_time, game_over):
        """计算僵尸进屋惩罚"""
        self.zombie_enter_count += 1
        punish = -10 * self.zombie_enter_count  # 梯度惩罚：-10, -20, -30...

        # 检查是否在5秒内连续进屋
        if current_time - self.last_zombie_enter_time < 5000:
            punish *= 1.5  # 5秒内再次进屋加重惩罚

        self.last_zombie_enter_time = current_time
        self.current_reward += punish

        # 游戏结束时追加总惩罚
        if game_over:
            final_punish = -self.current_reward * 1.5
            self.current_reward += final_punish
            return punish + final_punish

        return punish

    def reset_reward_tracking(self):
        """重置奖励跟踪变量（每局开始时）"""
        self.current_reward = 0
        self.achieved_time_intervals = set()
        self.zombie_enter_count = 0
        self.last_zombie_enter_time = 0

    def normalize_reward(self, reward):
        """将奖励归一化到[-1, 1]范围"""
        min_r = -100  # 估计的最小奖励
        max_r = 50  # 估计的最大奖励
        return (reward - min_r) / (max_r - min_r) if (max_r - min_r) != 0 else 0