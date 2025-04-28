# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh
# Modified with Rainbow DQN components and updated training loop
# v3: Fixed distributional loss gather, added weight norm logging, adjusted noisy log

import torch
import torch.nn as nn
import torch.nn.functional as F # Import Functional for C51 loss
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import math # For Noisy Nets
from tqdm import tqdm

gym.register_envs(ale_py)

# Helper function to calculate average weight norm
def calculate_average_weight_norm(model):
    total_norm = 0.0
    num_weight_tensors = 0
    for param in model.parameters():
        if param.dim() > 1: # Only consider weight matrices/tensors, not biases/vectors
            total_norm += torch.norm(param.data, p=2).item()
            num_weight_tensors += 1
    if num_weight_tensors == 0:
        return 0.0
    return total_norm / num_weight_tensors

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # NoisyLinear will handle its own initialization

# --- Noisy Layer ---
class NoisyLinear(nn.Module):
    """ Factorised NoisyLayer with Gaussian noise """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)

        nn.init.constant_(self.weight_sigma, self.sigma_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # Factorized Gaussian noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out) # Use scaled noise for bias too

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            # Generate noise only if training
            weight_epsilon = torch.randn(self.out_features, self.in_features, device=self.weight_mu.device)
            bias_epsilon = torch.randn(self.out_features, device=self.bias_mu.device)
            # Use factorized noise generation during training if desired (original implementation)
            # self.reset_noise() # Or use the factorized version by uncommenting this and commenting above 2 lines

            weight = self.weight_mu + self.weight_sigma * weight_epsilon # Use non-factorized noise if using randn directly
            bias = self.bias_mu + self.bias_sigma * bias_epsilon
        else:
            # Use mean weights during evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class DQN(nn.Module):
    """
        Rainbow DQN Network Architecture
    """
    def __init__(self, num_actions, args):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.use_dueling = args.use_dueling
        self.use_noisy = args.use_noisy
        self.use_distributional = args.use_distributional
        self.num_atoms = args.num_atoms if self.use_distributional else 1
        self.v_min = args.v_min if self.use_distributional else None
        self.v_max = args.v_max if self.use_distributional else None
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(args.device) if self.use_distributional else None

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )

        dummy_input = torch.zeros(1, 4, 84, 84)
        conv_out_size = self._get_conv_out(dummy_input)

        LinearLayer = NoisyLinear if self.use_noisy else nn.Linear
        fc_input_dim = conv_out_size

        if self.use_dueling:
            self.value_fc = nn.Sequential(LinearLayer(fc_input_dim, 512), nn.ReLU())
            self.value_output = LinearLayer(512, self.num_atoms)
            self.advantage_fc = nn.Sequential(LinearLayer(fc_input_dim, 512), nn.ReLU())
            self.advantage_output = LinearLayer(512, self.num_actions * self.num_atoms)
        else:
            self.fc_layers = nn.Sequential(
                LinearLayer(fc_input_dim, 512), nn.ReLU(),
                LinearLayer(512, self.num_actions * self.num_atoms)
            )

        if not self.use_noisy:
             self.apply(init_weights)

    def _get_conv_out(self, shape):
        o = self.conv_layers(shape)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x / 255.0
        conv_out = self.conv_layers(x)
        batch_size = conv_out.size(0)
        fc_input = conv_out.view(batch_size, -1)

        if self.use_dueling:
            value = self.value_fc(fc_input)
            value = self.value_output(value)
            advantage = self.advantage_fc(fc_input)
            advantage = self.advantage_output(advantage)

            if self.use_distributional:
                 value = value.view(batch_size, 1, self.num_atoms)
                 advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)
                 q_logits = value + advantage - advantage.mean(dim=1, keepdim=True)
                 q_dist = F.softmax(q_logits, dim=2)
                 return q_dist, q_logits
            else:
                 advantage = advantage.view(batch_size, self.num_actions)
                 value = value.view(batch_size, 1)
                 q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
                 return q_values
        else:
            output = self.fc_layers(fc_input)
            if self.use_distributional:
                q_logits = output.view(batch_size, self.num_actions, self.num_atoms)
                q_dist = F.softmax(q_logits, dim=2)
                return q_dist, q_logits
            else:
                q_values = output
                return q_values

    def reset_noise(self):
        # Note: Resetting noise might happen per forward pass in NoisyLinear if implemented that way
        # Or could be called episodically here if needed by the specific NoisyLinear implementation.
        # The current NoisyLinear resets noise internally during forward pass if training.
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                     # Depending on implementation, might need module.reset_noise()
                     pass # Current NoisyLinear resets in forward

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
             gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif len(obs.shape) == 2:
             gray = obs
        else:
             raise ValueError("Unexpected observation shape: {}".format(obs.shape))
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return states, actions, rewards, next_states, dones, np.ones(batch_size), list(range(batch_size))

    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done):
        priority = self.max_priority
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        # Add small epsilon to prevent zero probabilities if all priorities are zero
        probs += 1e-6
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[i] for i in indices]

        beta = self.beta_by_frame(self.frame_idx)
        self.frame_idx += batch_size

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        epsilon = 1e-5
        for idx, prio in zip(batch_indices, batch_priorities):
            priority_value = max(abs(prio) + epsilon, epsilon)
            self.priorities[idx] = priority_value
            self.max_priority = max(self.max_priority, priority_value)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env_name = env_name
        self.env = gym.make(env_name, frameskip=args.frame_skip)
        # Apply seed to environment's action space for reproducibility if needed
        self.env.action_space.seed(args.seed)
        self.test_env = gym.make(env_name, frameskip=args.frame_skip)

        self.num_actions = self.env.action_space.n
        self.args = args
        self.preprocessor = AtariPreprocessor(frame_stack=4)

        self.device = args.device
        print("Using device:", self.device)

        self.q_net = DQN(self.num_actions, args).to(self.device)
        self.target_net = DQN(self.num_actions, args).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4)

        if args.use_per:
            self.memory = PrioritizedReplayBuffer(args.memory_size, args.per_alpha, args.per_beta_start, args.per_beta_frames)
        else:
            self.memory = ReplayBuffer(args.memory_size)

        self.use_multistep = args.use_multistep
        self.n_steps = args.n_steps
        if self.use_multistep:
            self.n_step_buffer = deque(maxlen=self.n_steps)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        # Epsilon is only relevant if not using noisy nets
        self.epsilon = args.epsilon_start if not args.use_noisy else 0.0
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -float('inf')
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_frequency = args.train_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.support = self.q_net.support
        self.delta_z = (args.v_max - args.v_min) / (args.num_atoms - 1) if args.use_distributional else None

    def select_action(self, state):
        # Noisy Nets handle exploration during training implicitly
        if self.args.use_noisy and self.q_net.training:
             # No need for torch.no_grad() if NoisyLinear handles noise generation internally
             state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
             # Force network to be in train mode for noise generation if needed by NoisyLinear implementation
             # self.q_net.train() # Should already be in train mode during run loop
             if self.args.use_distributional:
                 dist, _ = self.q_net(state_tensor)
                 q_values = (dist * self.support).sum(2)
             else:
                 q_values = self.q_net(state_tensor)
             return q_values.argmax().item()

        # Epsilon-greedy exploration (only if not using Noisy Nets)
        if not self.args.use_noisy:
            eps_threshold = self.epsilon_min + \
                            (self.args.epsilon_start - self.epsilon_min) * \
                            max(0.0, 1.0 - self.env_count / self.args.epsilon_final_frame)
            self.epsilon = eps_threshold # Update for logging

            if random.random() < eps_threshold:
                return random.randint(0, self.num_actions - 1)

        # Exploitation (either epsilon-greedy didn't trigger, or using noisy nets in eval mode)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # Ensure network is in eval for deterministic output if not training (or if using Noisy in eval)
        current_mode_is_training = self.q_net.training
        if not current_mode_is_training:
             self.q_net.eval()

        with torch.no_grad():
             if self.args.use_distributional:
                 dist, _ = self.q_net(state_tensor)
                 q_values = (dist * self.support).sum(2)
             else:
                 q_values = self.q_net(state_tensor)

        # Restore mode if we changed it
        if not current_mode_is_training:
             self.q_net.train()

        return q_values.argmax().item()

    def _store_transition(self, state, action, reward, next_state, done):
        if not self.use_multistep:
             if isinstance(self.memory, PrioritizedReplayBuffer):
                 self.memory.add(state, action, reward, next_state, done)
             else:
                 self.memory.add(state, action, reward, next_state, done)
             return

        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_steps:
            return

        R = 0.0
        n_state, n_action, _, _, _ = self.n_step_buffer[0]
        # Get the nth next_state and done correctly
        nth_s, nth_a, nth_r, nth_ns, nth_d = self.n_step_buffer[-1]
        n_next_state, n_done = nth_ns, nth_d

        for i in range(self.n_steps):
            s, a, r, ns, d = self.n_step_buffer[i]
            R += (self.gamma ** i) * r
            if d:
                 # If terminated within N steps, use that state and done flag
                 n_next_state = ns
                 n_done = True
                 break # Stop accumulating reward for this transition

        # Store the N-step transition (s_t, a_t, R_t^{N}, s_{t+N}, done_{t+N})
        if isinstance(self.memory, PrioritizedReplayBuffer):
             self.memory.add(n_state, n_action, R, n_next_state, n_done)
        else:
             self.memory.add(n_state, n_action, R, n_next_state, n_done)


    def run(self):
        print(f"Starting training for {self.args.max_steps} environment steps...")
        episode = 0
        pbar = tqdm(total=self.args.max_steps, desc="Training Progress")
        last_log_env_count = 0 # Track steps for progress bar update

        while self.env_count < self.args.max_steps:
            episode += 1
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            episode_reward = 0
            episode_steps = 0

            self.q_net.train() # Ensure network is in training mode at episode start
            # Noisy nets don't require episodic noise reset here if reset in forward pass

            while not done and episode_steps < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                clipped_reward = np.sign(reward) # Use clipped reward for storage/training
                next_state = self.preprocessor.step(next_obs)

                self._store_transition(state, action, clipped_reward, next_state, done)

                state = next_state
                episode_reward += reward # Accumulate unclipped reward for logging
                self.env_count += 1
                episode_steps += 1

                # Update progress bar more efficiently
                steps_since_last_log = self.env_count - last_log_env_count
                if steps_since_last_log > 0:
                    pbar.update(steps_since_last_log)
                    last_log_env_count = self.env_count


                # --- Train the Network ---
                if len(self.memory) >= self.replay_start_size and self.env_count % self.train_frequency == 0:
                    for _ in range(self.train_per_step):
                         if len(self.memory) >= self.batch_size:
                             self.train()

                # --- Update Target Network ---
                if self.train_count > 0 and self.train_count % self.target_update_frequency == 0:
                    print(f"\n--- Updating target network at train step {self.train_count} (Env Step: {self.env_count}) ---")
                    self.target_net.load_state_dict(self.q_net.state_dict())

                # --- Evaluate and Save Model ---
                if self.env_count > 0 and self.env_count % 10000 == 0:
                    # Calculate weight norm before evaluation potentially changes network mode
                    avg_norm = calculate_average_weight_norm(self.q_net)
                    print(f"\n--- Evaluating and Saving at Env Step {self.env_count} ---")
                    eval_reward = self.evaluate()
                    print(f"[Eval @ {self.env_count} Steps] Avg Reward (5 eps): {eval_reward:.2f} | Avg Weight Norm: {avg_norm:.4f} | Train Steps: {self.train_count}")

                    log_dict = {
                        "Eval Reward": eval_reward,
                        "Avg Weight Norm": avg_norm, # Log the calculated norm
                        "Total Env Steps": self.env_count,
                        "Train Steps": self.train_count,
                    }
                    # Only log PER Beta if PER is actually used
                    if self.args.use_per:
                        current_beta = self.memory.beta_by_frame(self.memory.frame_idx)
                        log_dict["PER Beta"] = current_beta
                    wandb.log(log_dict, step=self.env_count)

                    model_path = os.path.join(self.save_dir, f"model_steps_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved model checkpoint to {model_path}")

                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        best_model_path = os.path.join(self.save_dir, "best_model.pt")
                        torch.save(self.q_net.state_dict(), best_model_path)
                        print(f"Saved new best model to {best_model_path} with reward {eval_reward:.2f}")
                    print("---------------------------------------------------\n")

                if self.env_count >= self.args.max_steps:
                    print(f"Reached max_steps ({self.args.max_steps}). Terminating training.")
                    break

            # Log end of episode info
            # Adjust Epsilon logging based on whether Noisy Nets are used
            exploration_info = f"Eps: {self.epsilon:.3f}" if not self.args.use_noisy else "Expl: Noisy"
            print(f"[Ep {episode}] Reward: {episode_reward:.2f} Steps: {episode_steps} TotalEnvSteps: {self.env_count}/{self.args.max_steps} TrainSteps: {self.train_count} {exploration_info}")
            log_data = {
                "Episode": episode,
                "Episode Reward (Unclipped)": episode_reward,
                "Episode Steps": episode_steps,
                "Memory Size": len(self.memory),
                "Total Env Steps": self.env_count,
                "Train Steps": self.train_count,
            }
            # Only log epsilon if not using noisy nets
            if not self.args.use_noisy:
                log_data["Epsilon"] = self.epsilon
            wandb.log(log_data, step=self.env_count)


        pbar.close()
        print("Training finished.")


    def evaluate(self, num_episodes=5):
        total_rewards = []
        self.q_net.eval() # Set Q-network to evaluation mode

        # print(f"Running evaluation for {num_episodes} episodes...") # Reduced verbosity
        for i in range(num_episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            episode_reward = 0
            step_count = 0
            max_eval_steps = 27000

            while not done and step_count < max_eval_steps:
                 # Use select_action which handles eval mode for Noisy Nets correctly
                 action = self.select_action(state) # Will use deterministic weights if noisy
                 next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                 done = terminated or truncated
                 next_state = self.preprocessor.step(next_obs)
                 state = next_state
                 episode_reward += reward
                 step_count += 1
            total_rewards.append(episode_reward)
            # print(f"  Eval Episode {i+1}/{num_episodes} Reward: {episode_reward:.2f}") # Reduced verbosity

        self.q_net.train() # Set Q-network back to training mode
        avg_reward = np.mean(total_rewards)
        return avg_reward


    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones, weights, indices = batch

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device) # Shape: [B]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        if self.args.use_per:
             weights = torch.from_numpy(weights).float().to(self.device)

        if self.args.use_distributional:
            with torch.no_grad():
                if self.args.use_double_dqn:
                    next_q_dist_online, _ = self.q_net(next_states)
                    next_q_values_online = (next_q_dist_online * self.support).sum(2)
                    next_actions = next_q_values_online.argmax(1) # Shape: [B]
                    next_q_dist_target, _ = self.target_net(next_states) # Shape: [B, A, N_atoms]
                    # Gather target distribution for next actions selected by online net
                    next_dist = next_q_dist_target[range(self.batch_size), next_actions] # Shape: [B, N_atoms]
                else:
                    next_q_dist_target, _ = self.target_net(next_states) # Shape: [B, A, N_atoms]
                    next_q_values_target = (next_q_dist_target * self.support).sum(2)
                    next_actions = next_q_values_target.argmax(1) # Shape: [B]
                    next_dist = next_q_dist_target[range(self.batch_size), next_actions] # Shape: [B, N_atoms]

                gamma = self.gamma ** self.n_steps if self.use_multistep else self.gamma
                Tz = rewards.unsqueeze(1) + gamma * self.support.unsqueeze(0) * (1 - dones.unsqueeze(1)) # Shape: [B, N_atoms]
                Tz = Tz.clamp(self.args.v_min, self.args.v_max)

                b = (Tz - self.args.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()

                # Fix edge cases: prevent indices from going out of bounds [0, N_atoms-1]
                l = torch.max(l, torch.zeros_like(l)) # Ensure l >= 0
                u = torch.min(u, torch.full_like(u, self.args.num_atoms - 1)) # Ensure u <= N_atoms - 1
                l[(u > 0) * (l == u)] -= 1 # Handle l==u cases except at boundaries
                u[(l < (self.args.num_atoms - 1)) * (l == u)] += 1 # Handle l==u cases except at boundaries
                l = torch.max(l, torch.zeros_like(l)) # Re-ensure l >= 0 after adjustment
                u = torch.min(u, torch.full_like(u, self.args.num_atoms - 1)) # Re-ensure u <= N_atoms - 1 after adjustment


                m = torch.zeros(self.batch_size, self.args.num_atoms, device=self.device)
                offset = torch.linspace(0, (self.batch_size - 1) * self.args.num_atoms, self.batch_size).long()\
                           .unsqueeze(1).expand(self.batch_size, self.args.num_atoms).to(self.device)

                # Project probability mass
                m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

                target_dist = m # Shape: [B, N_atoms]

            # --- Corrected Log Probability Calculation ---
            current_dist, current_logits = self.q_net(states) # current_dist shape: [B, A, N_atoms]
            # actions shape: [B] -> need [B, 1, 1] to gather along dim 1
            action_idx = actions.view(-1, 1, 1).expand(-1, 1, self.args.num_atoms) # Shape: [B, 1, N_atoms]
            # Gather the probability distribution for the chosen action
            chosen_action_dist = current_dist.gather(1, action_idx).squeeze(1) # Shape: [B, N_atoms]

            # Add small epsilon for numerical stability before log
            log_p = torch.log(chosen_action_dist + 1e-6) # Shape: [B, N_atoms]
            # --- End Correction ---

            # Calculate cross-entropy loss: L = - sum(target_dist * log_p) over atoms
            loss = -(target_dist * log_p).sum(1) # Shape: [B]
            td_errors = loss.detach().abs().cpu().numpy() # Use absolute error for priority

        else: # Standard DQN Loss Calculation
            q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                if self.args.use_double_dqn:
                     next_q_values_online = self.q_net(next_states)
                     best_actions = next_q_values_online.argmax(1).unsqueeze(1)
                     next_q_values_target = self.target_net(next_states).gather(1, best_actions).squeeze(1)
                else:
                     next_q_values_target = self.target_net(next_states).max(1)[0]

                gamma = self.gamma ** self.n_steps if self.use_multistep else self.gamma
                expected_q_values = rewards + gamma * next_q_values_target * (1 - dones)

            loss = F.smooth_l1_loss(q_values, expected_q_values, reduction='none')
            td_errors = loss.detach().abs().cpu().numpy() # Use absolute error for priority


        # Apply Importance Sampling Weights (if using PER) and calculate mean loss
        if self.args.use_per:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()

        return loss, td_errors


    def train(self):
        self.train_count += 1
        self.q_net.train()

        if self.args.use_per:
             states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)
             batch = (states, actions, rewards, next_states, dones, weights, indices)
        else:
             states, actions, rewards, next_states, dones, _, _ = self.memory.sample(self.batch_size)
             batch = (states, actions, rewards, next_states, dones, np.ones(self.batch_size), list(range(self.batch_size)))

        loss, td_errors = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.args.clip_grad_norm)
        self.optimizer.step()

        if self.args.use_per:
            self.memory.update_priorities(indices, td_errors) # Pass absolute td_errors

        # Log training loss periodically
        if self.train_count % 1000 == 0:
             wandb.log({"Training Loss": loss.item()}, step=self.env_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rainbow DQN for Atari")

    # --- Environment and Hardware ---
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="Gym environment name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # --- Training Hyperparameters ---
    parser.add_argument("--max-steps", type=int, default=1000000, help="Maximum number of environment steps to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate") # 6.25e-5
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--memory-size", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--replay-start-size", type=int, default=20000, help="Steps to fill buffer before training")
    parser.add_argument("--discount-factor", "--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target-update-frequency", type=int, default=8000, help="Frequency (in train steps) to update target network")
    parser.add_argument("--train-frequency", type=int, default=1, help="Frequency (in env steps) to train")
    parser.add_argument("--train-per-step", type=int, default=4, help="Training updates per training step")
    parser.add_argument("--max-episode-steps", type=int, default=27000, help="Maximum steps per episode")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon (if not using Noisy Nets)")
    parser.add_argument("--epsilon-final-frame", type=int, default=100000, help="Frame to anneal epsilon over (if not using Noisy Nets)")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Final epsilon value (if not using Noisy Nets)")
    parser.add_argument("--clip-grad-norm", type=float, default=10.0, help="Gradient clipping norm")
    parser.add_argument("--frame-skip", type=int, default=4, help="Frame skip")

    # --- Rainbow Component Flags ---
    # Set defaults according to user request
    parser.add_argument("--use-double-dqn", action='store_true', default=False, help="Enable Double DQN")
    parser.add_argument("--use-per", action='store_true', default=False, help="Enable Prioritized Experience Replay")
    parser.add_argument("--use-dueling", action='store_true', default=False, help="Enable Dueling Network")
    parser.add_argument("--use-multistep", action='store_true', default=False, help="Enable Multi-step Learning")
    parser.add_argument("--use-distributional", action='store_true', default=False, help="Enable Distributional RL (C51)") # DEFAULT TRUE
    parser.add_argument("--use-noisy", action='store_true', default=False, help="Enable Noisy Nets") # DEFAULT TRUE

    # --- Component Specific Parameters ---
    parser.add_argument("--per-alpha", type=float, default=0.5, help="Alpha for PER")
    parser.add_argument("--per-beta-start", type=float, default=0.4, help="Initial beta for PER")
    parser.add_argument("--per-beta-frames", type=int, default=1000000, help="Frames to anneal beta for PER")
    parser.add_argument("--n-steps", type=int, default=3, help="N for Multi-step Learning")
    parser.add_argument("--num-atoms", type=int, default=51, help="Atoms for Distributional RL")
    parser.add_argument("--v-min", type=float, default=-10.0, help="Min value for Distributional RL")
    parser.add_argument("--v-max", type=float, default=10.0, help="Max value for Distributional RL")

    # --- Logging and Saving ---
    parser.add_argument("--save-dir", type=str, default="./results_rainbow", help="Directory to save models")
    parser.add_argument("--wandb-project-name", type=str, default="DLP-Lab5-Rainbow-Pong", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

    args = parser.parse_args()

    # --- Set Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Set seed for environment action space as well for consistency
    # Note: Environment reset seed is handled implicitly by gym if seed is passed to make
    # However, action_space seed needs separate setting if full determinism is desired.
    # Let's set it inside the agent init for clarity.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # --- Create Run Name ---
    if args.wandb_run_name is None:
        run_name = f"{args.env_name.split('/')[-1]}"
        flags = []
        if args.use_double_dqn: flags.append("Double")
        if args.use_per: flags.append("PER")
        if args.use_dueling: flags.append("Dueling")
        if args.use_multistep: flags.append(f"Multi{args.n_steps}")
        if args.use_distributional: flags.append("Dist")
        if args.use_noisy: flags.append("Noisy")
        if not flags: flags.append("Vanilla")
        run_name += f"-{'-'.join(flags)}"
        run_name += f"-{time.strftime('%Y%m%d-%H%M%S')}"
        args.wandb_run_name = run_name

    # --- Initialize W&B ---
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args, save_code=True)

    print("Starting run with configuration:")
    for key, val in vars(args).items():
        print(f"  {key}: {val}")

    # --- Initialize Agent and Run ---
    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run()

    wandb.finish()
    print("Run finished.")