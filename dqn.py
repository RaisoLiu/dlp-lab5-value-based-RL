# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
import wandb
import argparse
import time
import yaml
from collections import deque
from tqdm import tqdm


gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DuelingDQN(nn.Module):
    def __init__(self, num_actions):
        super(DuelingDQN, self).__init__()
        self.num_actions = num_actions

        # 共享的卷積層
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        conv_output_size = 64 * 7 * 7 # 3136

        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512), nn.ReLU(),
            nn.Linear(512, 1) # 輸出 V(s)
        )

        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512), nn.ReLU(),
            nn.Linear(512, num_actions) # 輸出 A(s, a)
        )

    def forward(self, x):
        # 首先通過共享的卷積層
        features = self.conv_layers(x / 255.0) # 歸一化

        # 分別計算 V(s) 和 A(s, a)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # 組合 V 和 A -> Q
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        q_values = values + (advantages - advantages.mean(1, keepdim=True))

        return q_values


class FineGrainedDuelingDQN(nn.Module):
    """
    A Dueling DQN with a potentially finer-grained CNN architecture.
    """
    def __init__(self, num_actions):
        super(FineGrainedDuelingDQN, self).__init__()
        self.num_actions = num_actions

        # 更精細的共享卷積層
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2), nn.ReLU(),  # Smaller kernel, stride 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2), nn.ReLU(), # Added a layer
            nn.Conv2d(128, 128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        # Need to calculate the output size dynamically or pre-calculate it
        # For input (4, 84, 84):
        # (84 - 5) / 2 + 1 = 40.5 -> 40
        # (40 - 3) / 2 + 1 = 19.5 -> 19
        # (19 - 3) / 2 + 1 =  8.5 -> 9 (Mistake in thought, should be 8, let's recheck common padding/output size formulas or test)
        # Let's re-calculate assuming default padding=0:
        # Conv1: (84 - 5) // 2 + 1 = 79 // 2 + 1 = 39 + 1 = 40. Output: (32, 40, 40)
        # Conv2: (40 - 3) // 2 + 1 = 37 // 2 + 1 = 18 + 1 = 19. Output: (64, 19, 19)
        # Conv3: (19 - 3) // 2 + 1 = 16 // 2 + 1 = 8 + 1 = 9.  Output: (128, 9, 9)
        # Conv4: (9 - 3) // 1 + 1 = 6 // 1 + 1 = 7.          Output: (128, 7, 7)
        conv_output_size = 128 * 7 * 7 # 6272

        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512), nn.ReLU(),
            nn.Linear(512, 1) # 輸出 V(s)
        )

        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512), nn.ReLU(),
            nn.Linear(512, num_actions) # 輸出 A(s, a)
        )
        self.apply(init_weights) # Apply weight initialization


    def forward(self, x):
        # 首先通過共享的卷積層
        features = self.conv_layers(x / 255.0) # 歸一化

        # 分別計算 V(s) 和 A(s, a)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # 組合 V 和 A -> Q
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        q_values = values + (advantages - advantages.mean(1, keepdim=True))

        return q_values


class DQN(nn.Module):
    def __init__(self, env, num_actions):
        super(DQN, self).__init__()
        self.env = env

        if env == "CartPole-v1":
            self.network = nn.Sequential(
                nn.Linear(4, 128), # Increased hidden layer size
                nn.ReLU(),
                nn.Linear(128, 128), # Increased hidden layer size
                nn.ReLU(),
                nn.Linear(128, num_actions)
            )
        elif env == "ALE/Pong-v5":
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        self.apply(init_weights) # Apply weight initialization

    def forward(self, x):
        # CartPole state doesn't need normalization like image data
        # print(f"x.shape: {x.shape}")
        if self.env == "CartPole-v1":
            return self.network(x)
        elif self.env == "ALE/Pong-v5":
            return self.network(x / 255.0)


# FIXED
# class PrioritizedReplayBuffer:
#     """
#         Prioritizing the samples in the replay memory by the Bellman error
#         See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
#     """ 
#     def __init__(self, capacity, alpha=0.6, beta=0.4):
#         self.capacity = capacity
#         self.alpha = alpha
#         self.beta = beta
#         self.buffer = []
#         self.priorities = np.zeros((capacity,), dtype=np.float32)
#         self.pos = 0

#     def add(self, transition, error):
#         ########## YOUR CODE HERE (for Task 3) ########## 
                    
#         ########## END OF YOUR CODE (for Task 3) ########## 
#         return 
#     def sample(self, batch_size):
#         ########## YOUR CODE HERE (for Task 3) ########## 
                    
#         ########## END OF YOUR CODE (for Task 3) ########## 
#         return
#     def update_priorities(self, indices, errors):
#         ########## YOUR CODE HERE (for Task 3) ########## 
                    
#         ########## END OF YOUR CODE (for Task 3) ########## 
#         return
import numpy as np
import random

class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        # print(f"obs.shape: {obs.shape}")
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # print(f"resized.shape: {resized.shape}")
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class ReplayBuffer:
    """
        A simple FIFO replay buffer
    """
    def __init__(self, capacity, n_step=1, gamma=0.99):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = []

    def clean(self):
        """清空 n_step_buffer"""
        self.n_step_buffer = []

    def add(self, transition, error=None):
        # 將新的轉換添加到 n_step_buffer
        self.n_step_buffer.append(transition)
        
        # 如果 n_step_buffer 已滿，計算 n-step 獎勵並存儲
        if len(self.n_step_buffer) >= self.n_step:
            # 計算 n-step 獎勵
            n_step_reward = 0
            for i in range(self.n_step):
                n_step_reward += self.gamma ** i * self.n_step_buffer[i][2]  # 2 是 reward 的位置
            
            # 創建新的轉換元組
            n_step_transition = (
                self.n_step_buffer[0][0],  # 初始狀態
                self.n_step_buffer[0][1],  # 初始動作
                n_step_reward,             # n-step 獎勵
                self.n_step_buffer[-1][3], # 最終狀態
                self.n_step_buffer[-1][4]  # 最終 done 標誌
            )
            
            # 存儲到主緩衝區
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.pos] = n_step_transition
            self.pos = (self.pos + 1) % self.capacity
            
            # 移除最舊的轉換
            self.n_step_buffer.pop(0)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return [], [], []
        batch = random.sample(self.buffer, batch_size)
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        weights = np.ones(batch_size, dtype=np.float32)
        return batch, indices, weights

    def update_priorities(self, indices, errors):
        pass

    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=1e-5, n_step=1, gamma=0.99):
        """
        Initialize the Prioritized Replay Buffer.

        Args:
            capacity (int): Maximum number of experiences to store.
            alpha (float): Controls the degree of prioritization. alpha=0 means uniform sampling. [cite: 51]
            beta (float): Controls the degree of importance sampling correction. beta=1 means full correction. [cite: 52]
            beta_increment_per_sampling (float): Amount to increase beta each time sample() is called.
            epsilon (float): Small constant added to priorities to ensure non-zero probability. [cite: 50]
            n_step (int): Number of steps for N-step returns.
            gamma (float): Discount factor for N-step returns.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling # For beta annealing
        self.epsilon = epsilon  # Small amount to avoid zero priority
        self.buffer = [None] * capacity # Initialize buffer with None
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = []

    def clean(self):
        """清空 n_step_buffer"""
        self.n_step_buffer = []

    def add(self, transition, error):
        """
        Adds a new experience transition to the buffer along with its initial priority.

        Args:
            transition (tuple): The experience tuple (state, action, reward, next_state, done).
            error (float): The TD error (Bellman error) for this transition.
        """
        # 將新的轉換添加到 n_step_buffer
        self.n_step_buffer.append(transition)
        
        # 如果 n_step_buffer 已滿，計算 n-step 獎勵並存儲
        if len(self.n_step_buffer) >= self.n_step:
            # 計算 n-step 獎勵
            n_step_reward = 0
            for i in range(self.n_step):
                n_step_reward += self.gamma ** i * self.n_step_buffer[i][2]  # 2 是 reward 的位置
            
            # 創建新的轉換元組
            n_step_transition = (
                self.n_step_buffer[0][0],  # 初始狀態
                self.n_step_buffer[0][1],  # 初始動作
                n_step_reward,             # n-step 獎勵
                self.n_step_buffer[-1][3], # 最終狀態
                self.n_step_buffer[-1][4]  # 最終 done 標誌
            )
            
            # 計算優先級
            priority = abs(error) + self.epsilon
            max_priority = np.max(self.priorities) if self.size > 0 else 1.0
            if max_priority == 0:
                max_priority = 1.0
            
            # 存儲到主緩衝區
            self.buffer[self.pos] = n_step_transition
            self.priorities[self.pos] = max_priority
            
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            
            # 移除最舊的轉換
            self.n_step_buffer.pop(0)

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer based on their priorities.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing:
                - batch (list): A list of sampled experience transitions.
                - indices (np.ndarray): An array of indices corresponding to the sampled transitions.
                - weights (np.ndarray): An array of importance sampling weights for the sampled transitions.
        """
        ########## YOUR CODE HERE (for Task 3) ##########
        if self.size == 0:
            return [], [], [] # Return empty if buffer is empty

        # Get priorities of currently stored experiences
        priorities_slice = self.priorities[:self.size]

        # Calculate sampling probabilities: P(i) = p_i^alpha / sum(p_k^alpha) [cite: 51]
        scaled_priorities = priorities_slice ** self.alpha
        prob_distribution = scaled_priorities / np.sum(scaled_priorities)

        # Sample indices based on the probability distribution
        indices = np.random.choice(self.size, batch_size, p=prob_distribution, replace=True)

        # Retrieve the sampled transitions
        batch = [self.buffer[i] for i in indices]

        # Calculate Importance Sampling (IS) weights: w_i = (1 / (N * P(i)))^beta [cite: 52]
        weights = (self.size * prob_distribution[indices]) ** (-self.beta)
        # Normalize weights for stability: w_i = w_i / max(w_j) [cite: 53]
        weights /= np.max(weights)
        weights = np.array(weights, dtype=np.float32) # Ensure correct dtype

        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        ########## END OF YOUR CODE (for Task 3) ##########
        return batch, indices, weights

    def update_priorities(self, indices, errors):
        """
        Updates the priorities of sampled experiences based on their new TD errors.

        Args:
            indices (np.ndarray): The indices of the experiences whose priorities need updating.
            errors (np.ndarray): The new TD errors corresponding to the indices.
        """
        ########## YOUR CODE HERE (for Task 3) ##########
        # Calculate new priorities: p = |error| + epsilon [cite: 47, 50]
        priorities = np.abs(errors) + self.epsilon
        # Update priorities in the array
        # Ensure indices are within the valid range of the current buffer size
        valid_indices_mask = indices < self.size
        valid_indices = indices[valid_indices_mask]
        valid_priorities = priorities[valid_indices_mask]

        if len(valid_indices) > 0:
            self.priorities[valid_indices] = valid_priorities
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def __len__(self):
        """Returns the current number of items in the buffer."""
        return self.size
        
# FIXED
class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env_name = env_name
        
        self.env = gym.make(env_name)#, render_mode="rgb_array")
        self.test_env = gym.make(env_name)#, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()
        self.dry = args.dry if hasattr(args, 'dry') else False
        self.use_ddqn = args.use_ddqn if hasattr(args, 'use_ddqn') else False
        self.use_dueling = args.use_dueling if hasattr(args, 'use_dueling') else False
        self.use_fine_cnn = args.use_fine_cnn if hasattr(args, 'use_fine_cnn') else False
        
        # 時間追蹤相關變數
        self.last_time = time.time()
        self.step_times = []
        self.last_checkpoint_step = 0
        
        # 根據參數選擇使用哪種 replay buffer
        if args.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(args.memory_size, n_step=args.n_step, gamma=args.gamma)
        else:
            self.memory = ReplayBuffer(args.memory_size, n_step=args.n_step, gamma=args.gamma)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        # 根據是否使用 DuelingDQN 選擇網路架構
        if self.env_name == "ALE/Pong-v5":
            if self.use_dueling:
                if self.use_fine_cnn:
                    print("Using Fine-Grained Dueling DQN for Pong")
                    self.q_net = FineGrainedDuelingDQN(self.num_actions).to(self.device)
                    self.target_net = FineGrainedDuelingDQN(self.num_actions).to(self.device)
                else:
                    print("Using Dueling DQN for Pong")
                    self.q_net = DuelingDQN(self.num_actions).to(self.device)
                    self.target_net = DuelingDQN(self.num_actions).to(self.device)
            else:
                 print("Using standard DQN for Pong")
                 self.q_net = DQN(env_name, self.num_actions).to(self.device)
                 self.target_net = DQN(env_name, self.num_actions).to(self.device)
        else: # For CartPole-v1 or other envs
             print(f"Using standard DQN for {env_name}")
             self.q_net = DQN(env_name, self.num_actions).to(self.device)
             self.target_net = DQN(env_name, self.num_actions).to(self.device)

        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = 1 if self.dry else args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_min = args.epsilon_min
        
        # 自動計算 epsilon_decay
        if hasattr(args, 'epsilon_decay') and args.epsilon_decay is not None:
            # 如果提供了 epsilon_decay，直接使用設定的值
            self.epsilon_decay = args.epsilon_decay
        elif hasattr(args, 'max_steps'):
            # 只有在沒有設定 epsilon_decay 的情況下，才根據 max_steps 計算
            # 計算需要多少步才能從 epsilon_start 降到 epsilon_min
            # epsilon = epsilon_start * (epsilon_decay)^steps
            # epsilon_min = epsilon_start * (epsilon_decay)^max_steps
            # epsilon_decay = (epsilon_min / epsilon_start)^(1/max_steps)
            self.epsilon_decay = (self.epsilon_min / self.epsilon) ** (1.0 / args.max_steps)
            args.epsilon_decay = self.epsilon_decay
        else:
            # 如果既沒有設定 epsilon_decay，也沒有設定 max_steps，使用預設值
            self.epsilon_decay = 0.9999
            args.epsilon_decay = self.epsilon_decay

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0 if self.env_name == "CartPole-v1" else -21 # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 新增：用於追蹤平均獎勵的滑動窗口
        self.reward_window = deque(maxlen=100)  # 追蹤最近100個episode的獎勵
        self.avg_reward = 0  # 當前平均獎勵
        self.eval_episodes = args.eval_episodes

                # 儲存超參數到 YAML 檔案
        if not self.dry:
            self._save_hyperparameters(args)

        # 如果提供了檢查點路徑，則載入檢查點
        if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
            self.load_checkpoint(args.checkpoint_path)

    def _save_hyperparameters(self, args):
        """
        將超參數儲存到 YAML 檔案中
        
        Args:
            args: 包含所有超參數的命名空間
        """
        hyperparams = {
            'env_name': self.env_name,
            'batch_size': args.batch_size,
            'memory_size': args.memory_size,
            'lr': args.lr,
            'discount_factor': args.discount_factor,
            'epsilon_start': args.epsilon_start,
            'epsilon_decay': args.epsilon_decay,
            'epsilon_min': args.epsilon_min,
            'target_update_frequency': args.target_update_frequency,
            'replay_start_size': args.replay_start_size,
            'max_episode_steps': args.max_episode_steps,
            'train_per_step': args.train_per_step,
            'use_prioritized_replay': args.use_prioritized_replay,
            'eval_episodes': args.eval_episodes,
            'use_dueling': args.use_dueling,
            'use_fine_cnn': args.use_fine_cnn
        }
        
        # 建立超參數檔案的路徑
        hyperparams_path = os.path.join(self.save_dir, 'hyperparameters.yaml')
        
        # 將超參數寫入 YAML 檔案
        with open(hyperparams_path, 'w') as f:
            yaml.dump(hyperparams, f, default_flow_style=False)
            
        print(f"超參數已儲存到 {hyperparams_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        從檢查點載入模型狀態
        
        Args:
            checkpoint_path (str): 檢查點檔案的路徑
        """
        if not os.path.exists(checkpoint_path):
            print(f"警告：找不到檢查點檔案 {checkpoint_path}")
            return
            

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint)
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"成功載入檢查點 {checkpoint_path}")


    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, max_steps=1000000):
        if self.dry:
            max_steps = 100
            print("Running in dry mode: only 100 steps")
        
        total_steps = 0
        ep = 0
        
        # 初始化 tqdm 進度條
        pbar = tqdm(total=max_steps, desc="Training Progress", unit="step")
        
        while total_steps < max_steps:
            obs, _ = self.env.reset()
            if "ALE/Pong-v5" in self.env_name:
                state = self.preprocessor.reset(obs)
            else:
                state = obs
            done = False
            total_reward = 0
            step_count = 0

            # 清空 n_step_buffer
            self.memory.clean()

            while not done and step_count < self.max_episode_steps and total_steps < max_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if "ALE/Pong-v5" in self.env_name:
                    next_state = self.preprocessor.step(next_obs)
                else:
                    next_state = next_obs
                self.memory.add((state, action, reward, next_state, done), 0)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                total_steps += 1
                
                # 更新進度條
                pbar.update(1)
                
                # 更新進度條描述
                if total_steps % 1000 == 0:
                    current_time = time.time()
                    step_time = current_time - self.last_time
                    self.step_times.append(step_time)
                    self.last_time = current_time
                    
                    # 計算預估剩餘時間
                    avg_time_per_step = sum(self.step_times) / len(self.step_times) if self.step_times else 0
                    remaining_steps = max_steps - total_steps
                    estimated_time_remaining = avg_time_per_step * remaining_steps / 1000  # 轉換為秒
                    
                    # 更新進度條描述
                    pbar.set_description(f"Training Progress (Est. time remaining: {estimated_time_remaining/60:.1f} min)")

                if total_steps % 10000 == 0 and not self.dry:
                    # 計算每 1000 步的平均時間
                    avg_time_per_1k = sum(self.step_times) / len(self.step_times) if self.step_times else 0
                    self.step_times = []  # 重置時間記錄
                    
                    # 儲存檢查點
                    model_path = os.path.join(self.save_dir, f"model_step{total_steps}.pt")
                    checkpoint = {
                        'model_state_dict': self.q_net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epsilon': self.epsilon,
                        'train_count': self.train_count,
                        'env_count': self.env_count,
                        'best_reward': self.best_reward
                    }
                    torch.save(checkpoint, model_path)
                    print(f"\nSaved model checkpoint to {model_path}")
                    
                    # 執行評估
                    eval_reward = self.evaluate()
                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        best_model_path = os.path.join(self.save_dir, f"best_model_step{total_steps}_r{eval_reward:.0f}.pt")
                        torch.save(checkpoint, best_model_path)
                        print(f"Saved new best model to {best_model_path} with reward {eval_reward}")
                    
                    # 記錄到 wandb
                    wandb.log({
                        "Total Steps": total_steps,
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "Total Reward": total_reward,
                        "Eval Reward": eval_reward,
                        "Time per 1k steps (s)": avg_time_per_1k
                    })
                    print(f"[Step {total_steps}] Episode: {ep}, Total Reward: {total_reward:.2f}, Eval Reward: {eval_reward:.2f}, Epsilon: {self.epsilon:.4f}, Time per 1k steps: {avg_time_per_1k:.2f}s")

            # 更新獎勵窗口和平均獎勵
            self.reward_window.append(total_reward)
            self.avg_reward = sum(self.reward_window) / len(self.reward_window)
            ep += 1
            
        # 關閉進度條
        pbar.close()

    def evaluate(self):
        total_reward_list = []
        for _ in range(self.eval_episodes):
            obs, _ = self.test_env.reset()
            if "ALE/Pong-v5" in self.env_name:
                state = self.preprocessor.reset(obs)
            else:
                state = obs
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                if "ALE/Pong-v5" in self.env_name:
                    state = self.preprocessor.step(next_obs)
                else:
                    state = next_obs

            total_reward_list.append(total_reward)

        return np.mean(total_reward_list)


    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilon-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        # 從優先級重放緩衝區中取樣
        batch, indices, weights = self.memory.sample(self.batch_size)
        if not batch:  # 如果批次為空，直接返回
            return
            
        # 解包批次數據
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 轉換為 PyTorch 張量
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        
        # 計算當前 Q 值
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # 計算目標 Q 值
        with torch.no_grad():
            if self.use_ddqn:
                # DDQN: 使用主網路選擇動作，目標網路評估 Q 值
                next_actions = self.q_net(next_states).max(1)[1]
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # DQN: 直接使用目標網路的最大 Q 值
                next_q_values = self.target_net(next_states).max(1)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # 計算 TD 誤差
        td_errors = (current_q_values.squeeze() - target_q_values).abs().detach().cpu().numpy()
        
        # 更新優先級
        self.memory.update_priorities(indices, td_errors)
        
        # 計算加權損失
        loss = (weights * (current_q_values.squeeze() - target_q_values).pow(2)).mean()
        
        # 優化步驟
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        
        # 計算梯度範數
        total_grad_norm = 0
        for p in self.q_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # 計算參數範數
        total_param_norm = 0
        for p in self.q_net.parameters():
            if p.requires_grad:
                param_norm = p.data.norm(2)
                total_param_norm += param_norm.item() ** 2
        total_param_norm = total_param_norm ** 0.5
        
        self.optimizer.step()
        
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        if self.train_count % 1000 == 0:
            if not self.dry:
                wandb.log({
                    "Train Loss": loss.item(),
                    "Q Value Mean": current_q_values.mean().item(),
                    "Q Value Std": current_q_values.std().item(),
                    "Q Value Min": current_q_values.min().item(),
                    "Q Value Max": current_q_values.max().item(),
                    "Gradient Norm": total_grad_norm,
                    "Parameter Norm": total_param_norm,
                    "Epsilon": self.epsilon
                })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", choices=["CartPole-v1", "ALE/Pong-v5"], help="Gym environment name")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float)
    parser.add_argument("--epsilon-min", type=float, default=0.1)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=10000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000000, help="Maximum number of steps to train")
    parser.add_argument("--use-prioritized-replay", action="store_true", help="Whether to use prioritized experience replay")
    parser.add_argument("--dry", action="store_true", help="Debug mode: run only 100 steps, no wandb logging")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--checkpoint-path", type=str, help="Path to checkpoint file")
    parser.add_argument("--use-ddqn", action="store_true", help="Whether to use Double DQN")
    parser.add_argument("--use-dueling", action="store_true", help="Whether to use Dueling DQN")
    parser.add_argument("--use-fine-cnn", action="store_true", help="Whether to use a finer-grained CNN architecture for Pong")
    parser.add_argument("--n-step", type=int, default=1, help="Number of steps for N-step returns")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for N-step returns")
    args = parser.parse_args()

    if not args.dry:
        wandb.init(project=f"DLP-Lab5-{args.env.replace('/', '-')}", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name=args.env, args=args)
    agent.run(max_steps=args.max_steps)

    ## Reference ##
    # https://ale.farama.org/environments/pong/
    # https://www.gymlibrary.dev/environments/classic_control/cart_pole/
