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


gym.register_envs(ale_py)


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
                nn.Linear(128, num_actions),
                nn.Softmax(dim=1)
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
                nn.Linear(512, num_actions),
                nn.Softmax(dim=1)
            )


    def forward(self, x):
        # CartPole state doesn't need normalization like image data
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

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    model = DQN(args.env, num_actions).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    total_reward_list = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        
        # 根據環境選擇適當的狀態處理方式
        if "ALE/Pong-v5" in args.env:
            state = preprocessor.reset(obs)
        else:
            state = obs
            
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # 根據環境選擇適當的狀態處理方式
            if "ALE/Pong-v5" in args.env:
                state = preprocessor.step(next_obs)
            else:
                state = next_obs
                
            frame_idx += 1

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)

        total_reward_list.append(total_reward)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")

    print(f"Average reward: {np.mean(total_reward_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    parser.add_argument("--env", type=str, default="CartPole-v1", choices=["CartPole-v1", "ALE/Pong-v5"], help="Gym environment name")
    args = parser.parse_args()
    evaluate(args)
