import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse
import yaml


gym.register_envs(ale_py)


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

    def forward(self, x):
        if self.env == "CartPole-v1":
            return self.network(x)
        elif self.env == "ALE/Pong-v5":
            return self.network(x / 255.0)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    is_atari = "ALE/" in args.env
    if is_atari:
        preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    # 嘗試載入超參數以確定是否使用 Dueling DQN
    hyperparams = {}
    hyperparams_path = os.path.join(os.path.dirname(args.model_path), 'hyperparameters.yaml')
    if os.path.exists(hyperparams_path):
        try:
            with open(hyperparams_path, 'r') as f:
                hyperparams = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load hyperparameters from {hyperparams_path}: {e}")

    # 根據超參數和環境選擇模型
    if hyperparams.get('use_dueling', False) and is_atari:
        print("Loading Dueling DQN model.")
        model = DuelingDQN(num_actions).to(device)
    else:
        print("Loading standard DQN model.")
        model = DQN(args.env, num_actions).to(device)

    # 載入模型權重
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
             model.load_state_dict(checkpoint['model_state_dict'])
        else:
             # 如果 checkpoint 只包含 state_dict
             model.load_state_dict(checkpoint)
        print(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        return # Or handle the error appropriately

    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    total_reward_list = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)

        # 根據環境選擇適當的狀態處理方式
        if is_atari:
            state = preprocessor.reset(obs)
        else:
            state = obs

        done = False
        total_reward = 0
        frames = []
        # frame_idx = 0 # Not used in the old version logic

        while not done:
            frame = env.render()
            frames.append(frame)

            # 轉換 state 為 tensor
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            else: # Assuming it might be a tuple or other structure for non-Atari envs initially
                 state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # 根據環境選擇適當的狀態更新方式
            if is_atari:
                state = preprocessor.step(next_obs)
            else:
                state = next_obs

            # frame_idx += 1 # Not used in the old version logic

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        try:
            with imageio.get_writer(out_path, fps=30) as video:
                for f in frames:
                    video.append_data(f)
            print(f"Saved episode {ep} with total reward {total_reward} -> {out_path}")
        except Exception as e:
            print(f"Error saving video for episode {ep}: {e}")


        total_reward_list.append(total_reward)


    if total_reward_list:
        print(f"Average reward over {args.episodes} episodes: {np.mean(total_reward_list):.2f}")
    else:
        print("No episodes were completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos", help="Directory to save evaluation videos")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    parser.add_argument("--env", type=str, default="CartPole-v1", choices=["CartPole-v1", "ALE/Pong-v5"], help="Gym environment name")
    args = parser.parse_args()

    evaluate(args)
