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
import torch.nn.functional as F


gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    def __init__(self, num_actions, use_distributional=False, atoms=51, use_noisy=False):
        super(DuelingDQN, self).__init__()
        self.num_actions = num_actions
        self.use_distributional = use_distributional
        self.atoms = atoms
        self.use_noisy = use_noisy

        # 共享的卷積層
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        conv_output_size = 64 * 7 * 7 # 3136

        # Value Stream
        if use_noisy:
            self.value_stream = nn.Sequential(
                NoisyLinear(conv_output_size, 512), nn.ReLU(),
                NoisyLinear(512, atoms if use_distributional else 1) # 輸出 V(s)
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(conv_output_size, 512), nn.ReLU(),
                nn.Linear(512, atoms if use_distributional else 1) # 輸出 V(s)
            )

        # Advantage Stream
        if use_noisy:
            self.advantage_stream = nn.Sequential(
                NoisyLinear(conv_output_size, 512), nn.ReLU(),
                NoisyLinear(512, num_actions * atoms if use_distributional else num_actions) # 輸出 A(s, a)
            )
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(conv_output_size, 512), nn.ReLU(),
                nn.Linear(512, num_actions * atoms if use_distributional else num_actions) # 輸出 A(s, a)
            )

    def reset_noise(self):
        if self.use_noisy:
            for module in self.value_stream:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
            for module in self.advantage_stream:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    def forward(self, x):
        # 首先通過共享的卷積層
        features = self.conv_layers(x / 255.0) # 歸一化

        # 分別計算 V(s) 和 A(s, a)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        if self.use_distributional:
            # 重塑優勢流輸出為 (batch_size, num_actions, atoms)
            advantages = advantages.view(-1, self.num_actions, self.atoms)
            # 重塑價值流輸出為 (batch_size, 1, atoms)
            values = values.view(-1, 1, self.atoms)
            
            # 組合 V 和 A -> Q
            # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
            q_values = values + (advantages - advantages.mean(1, keepdim=True))
            
            # 應用 softmax 以獲得機率分佈
            q_values = F.softmax(q_values, dim=2)
        else:
            # 標準 Dueling DQN 的組合方式
            values = values.expand(-1, self.num_actions)
            advantages = advantages - advantages.mean(1, keepdim=True)
            q_values = values + advantages

        return q_values


class FineGrainedDuelingDQN(nn.Module):
    """
    A Dueling DQN with a potentially finer-grained CNN architecture.
    """
    def __init__(self, num_actions, use_distributional=False, atoms=51, use_noisy=False):
        super(FineGrainedDuelingDQN, self).__init__()
        self.num_actions = num_actions
        self.use_distributional = use_distributional
        self.atoms = atoms
        self.use_noisy = use_noisy

        # 更精細的共享卷積層
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2), nn.ReLU(),  # Smaller kernel, stride 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2), nn.ReLU(), # Added a layer
            nn.Conv2d(128, 128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        conv_output_size = 128 * 7 * 7 # 6272

        # Value Stream
        if use_noisy:
            self.value_stream = nn.Sequential(
                NoisyLinear(conv_output_size, 512), nn.ReLU(),
                NoisyLinear(512, 1) # 輸出 V(s)
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(conv_output_size, 512), nn.ReLU(),
                nn.Linear(512, 1) # 輸出 V(s)
            )

        # Advantage Stream
        if use_noisy:
            self.advantage_stream = nn.Sequential(
                NoisyLinear(conv_output_size, 512), nn.ReLU(),
                NoisyLinear(512, num_actions) # 輸出 A(s, a)
            )
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(conv_output_size, 512), nn.ReLU(),
                nn.Linear(512, num_actions) # 輸出 A(s, a)
            )
        self.apply(init_weights)

    def reset_noise(self):
        if self.use_noisy:
            for module in self.value_stream:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
            for module in self.advantage_stream:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

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
    def __init__(self, env, num_actions, use_distributional=False, atoms=51, use_noisy=False):
        super(DQN, self).__init__()
        self.env = env
        self.use_distributional = use_distributional
        self.atoms = atoms
        self.use_noisy = use_noisy

        if env == "CartPole-v1":
            if use_noisy:
                self.network = nn.Sequential(
                    NoisyLinear(4, 128),
                    nn.ReLU(),
                    NoisyLinear(128, 128),
                    nn.ReLU(),
                    NoisyLinear(128, num_actions * atoms if use_distributional else num_actions)
                )
            else:
                self.network = nn.Sequential(
                    nn.Linear(4, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_actions * atoms if use_distributional else num_actions)
                )
        elif env == "ALE/Pong-v5":
            if use_noisy:
                self.network = nn.Sequential(
                    nn.Conv2d(4, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    NoisyLinear(64 * 7 * 7, 512),
                    nn.ReLU(),
                    NoisyLinear(512, num_actions * atoms if use_distributional else num_actions)
                )
            else:
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
                    nn.Linear(512, num_actions * atoms if use_distributional else num_actions)
                )
        self.apply(init_weights)

    def reset_noise(self):
        if self.use_noisy:
            for module in self.network:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    def forward(self, x):
        if self.env == "CartPole-v1":
            x = self.network(x)
        elif self.env == "ALE/Pong-v5":
            x = self.network(x / 255.0)
            
        if self.use_distributional:
            x = x.view(-1, self.num_actions, self.atoms)
            x = F.softmax(x, dim=2)
        return x


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
        self.use_distributional = args.use_distributional if hasattr(args, 'use_distributional') else False
        self.use_noisy = args.use_noisy if hasattr(args, 'use_noisy') else False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        # 設置 atoms 的預設值
        self.atoms = args.atoms if hasattr(args, 'atoms') else 51
        
        # 設置 v_min 和 v_max 的預設值
        self.v_min = args.v_min if hasattr(args, 'v_min') else -10.0
        self.v_max = args.v_max if hasattr(args, 'v_max') else 10.0
        
        # 設置 delta_z 的預設值
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        
        # 設置 support 的預設值
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms).to(self.device)
        
        # Distributional RL 相關參數
        if self.use_distributional:
            pass  # 所有必要的參數都已經設置好了
        
        # 時間追蹤相關變數
        self.last_time = time.time()
        self.step_times = []
        self.last_checkpoint_step = 0
        
        # 根據參數選擇使用哪種 replay buffer
        if args.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(args.memory_size, n_step=args.n_step, gamma=args.gamma)
        else:
            self.memory = ReplayBuffer(args.memory_size, n_step=args.n_step, gamma=args.gamma)
            

        # 根據是否使用 DuelingDQN 選擇網路架構
        if self.env_name == "ALE/Pong-v5":
            if self.use_dueling:
                if self.use_fine_cnn:
                    print("Using Fine-Grained Dueling DQN for Pong")
                    self.q_net = FineGrainedDuelingDQN(self.num_actions, self.use_distributional, self.atoms, self.use_noisy).to(self.device)
                    self.target_net = FineGrainedDuelingDQN(self.num_actions, self.use_distributional, self.atoms, self.use_noisy).to(self.device)
                else:
                    print("Using Dueling DQN for Pong")
                    self.q_net = DuelingDQN(self.num_actions, self.use_distributional, self.atoms, self.use_noisy).to(self.device)
                    self.target_net = DuelingDQN(self.num_actions, self.use_distributional, self.atoms, self.use_noisy).to(self.device)
            else:
                 print("Using standard DQN for Pong")
                 self.q_net = DQN(env_name, self.num_actions, self.use_distributional, self.atoms, self.use_noisy).to(self.device)
                 self.target_net = DQN(env_name, self.num_actions, self.use_distributional, self.atoms, self.use_noisy).to(self.device)
        else: # For CartPole-v1 or other envs
             print(f"Using standard DQN for {env_name}")
             self.q_net = DQN(env_name, self.num_actions, self.use_distributional, self.atoms, self.use_noisy).to(self.device)
             self.target_net = DQN(env_name, self.num_actions, self.use_distributional, self.atoms, self.use_noisy).to(self.device)

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
            'use_fine_cnn': args.use_fine_cnn,
            'use_distributional': args.use_distributional,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'atoms': self.atoms,
            'delta_z': self.delta_z,
            'support': self.support.tolist()
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
        if not self.use_noisy and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.use_distributional:
                # 對於分佈式 RL，我們需要計算每個動作的期望值
                q_dist = self.q_net(state_tensor)
                # 計算期望值：sum(p * z)，其中 z 是支援點
                q_values = (q_dist * self.support).sum(dim=2)
            else:
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
                    if self.use_distributional:
                        # 對於分佈式 RL，我們需要計算每個動作的期望值
                        q_dist = self.q_net(state_tensor)
                        # 計算期望值：sum(p * z)，其中 z 是支援點
                        q_values = (q_dist * self.support).sum(dim=2)
                    else:
                        q_values = self.q_net(state_tensor)
                    action = q_values.argmax().item()
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
        if not self.use_noisy and self.epsilon > self.epsilon_min:
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

        # 重置噪聲
        if self.use_noisy:
            self.q_net.reset_noise()
            self.target_net.reset_noise()

        if self.use_distributional:
            # 計算當前狀態分佈
            current_dist = self.q_net(states)
            current_dist = current_dist[range(self.batch_size), actions]
            
            # 計算下一個狀態分佈
            with torch.no_grad():
                next_dist = self.target_net(next_states)
                next_actions = next_dist.mean(2).max(1)[1]
                next_dist = next_dist[range(self.batch_size), next_actions]
                
                # 投影下一個狀態分佈
                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)
                support = self.support.unsqueeze(0)
                
                Tz = rewards + (1 - dones) * self.gamma * support
                Tz = Tz.clamp(self.v_min, self.v_max)
                b = (Tz - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()
                
                # 修復消失的機率質量
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.atoms - 1)) * (l == u)] += 1
                
                # 分配機率
                m = torch.zeros(self.batch_size, self.atoms, device=self.device)
                offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size, device=self.device).unsqueeze(1).expand(self.batch_size, self.atoms).long()
                m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
            
            # 計算交叉熵損失
            loss = -(m * current_dist.log()).sum(1).mean()
            
            # 計算 TD 誤差
            td_errors = (current_dist - m).abs().sum(1).detach().cpu().numpy()
        else:
            # 標準 DQN 損失計算
            current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
            
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
            
            # 計算加權損失
            loss = (weights * (current_q_values.squeeze() - target_q_values).pow(2)).mean()
        
        # 更新優先級
        self.memory.update_priorities(indices, td_errors)
        
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
                    "Q Value Mean": current_q_values.mean().item() if not self.use_distributional else (current_dist * self.support).sum(1).mean().item(),
                    "Q Value Std": current_q_values.std().item() if not self.use_distributional else (current_dist * self.support).sum(1).std().item(),
                    "Q Value Min": current_q_values.min().item() if not self.use_distributional else (current_dist * self.support).sum(1).min().item(),
                    "Q Value Max": current_q_values.max().item() if not self.use_distributional else (current_dist * self.support).sum(1).max().item(),
                    "Gradient Norm": total_grad_norm,
                    "Parameter Norm": total_param_norm,
                    "Epsilon": self.epsilon
                })

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        if self.use_distributional:
            # Get current state distribution
            current_dist = self.q_net(states)
            current_dist = current_dist[range(self.batch_size), actions]
            
            # Get next state distribution
            with torch.no_grad():
                next_dist = self.target_net(next_states)
                next_actions = next_dist.mean(2).max(1)[1]
                next_dist = next_dist[range(self.batch_size), next_actions]
                
                # Project next state distribution
                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)
                support = self.support.unsqueeze(0)
                
                Tz = rewards + (1 - dones) * self.gamma * support
                Tz = Tz.clamp(self.v_min, self.v_max)
                b = (Tz - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()
                
                # Fix disappearing probability mass
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.atoms - 1)) * (l == u)] += 1
                
                # Distribute probability
                m = torch.zeros(self.batch_size, self.atoms, device=self.device)
                offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).long()
                m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
            
            # Compute cross entropy loss
            loss = -(m * current_dist.log()).sum(1).mean()
        else:
            # Standard DQN loss
            current_q_values = self.q_net(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", choices=["CartPole-v1", "ALE/Pong-v5"], help="Gym environment name")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
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
    parser.add_argument("--use-distributional", action="store_true", help="Whether to use Distributional RL")
    parser.add_argument("--v-min", type=float, default=-10.0, help="Minimum value in the distributional RL support")
    parser.add_argument("--v-max", type=float, default=10.0, help="Maximum value in the distributional RL support")
    parser.add_argument("--atoms", type=int, default=51, help="Number of atoms in the distributional RL support")
    parser.add_argument("--use-noisy", action="store_true", help="Whether to use NoisyNet")
    args = parser.parse_args()

    if not args.dry:
        wandb.init(project=f"DLP-Lab5-{args.env.replace('/', '-')}", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name=args.env, args=args)
    agent.run(max_steps=args.max_steps)

    ## Reference ##
    # https://ale.farama.org/environments/pong/
    # https://www.gymlibrary.dev/environments/classic_control/cart_pole/
