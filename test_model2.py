# test_model2.py
# Adapted to load and evaluate models trained with dqn2.py (Rainbow DQN)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse
import math # For Noisy Nets

gym.register_envs(ale_py)

# --- Copied necessary components from dqn2.py ---

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # NoisyLinear will handle its own initialization

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
        self.reset_noise() # Initialize buffers

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)

        nn.init.constant_(self.weight_sigma, self.sigma_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        # Note: In eval mode, noise is not typically reset or used.
        # The forward pass uses mu parameters.
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        # Generate noise on the correct device during initialization/reset
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            # Generate noise using buffers if factorized, or sample directly
            # Using direct sampling as in dqn2.py's training forward pass logic:
            weight_epsilon = torch.randn_like(self.weight_epsilon)
            bias_epsilon = torch.randn_like(self.bias_epsilon)

            weight = self.weight_mu + self.weight_sigma * weight_epsilon
            bias = self.bias_mu + self.bias_sigma * bias_epsilon
        else:
            # Use mean weights during evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class DQN(nn.Module):
    """
        Rainbow DQN Network Architecture (Copied from dqn2.py)
    """
    def __init__(self, num_actions, args):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        # Get config from the passed args object
        self.use_dueling = args.use_dueling
        self.use_noisy = args.use_noisy
        self.use_distributional = args.use_distributional
        self.num_atoms = args.num_atoms if self.use_distributional else 1
        self.v_min = args.v_min if self.use_distributional else None
        self.v_max = args.v_max if self.use_distributional else None
        # Define support on the correct device specified in args
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
            self.value_output = LinearLayer(512, self.num_atoms) # Output atoms even if not distributional (will be size 1)
            self.advantage_fc = nn.Sequential(LinearLayer(fc_input_dim, 512), nn.ReLU())
            self.advantage_output = LinearLayer(512, self.num_actions * self.num_atoms)
        else:
            self.fc_layers = nn.Sequential(
                LinearLayer(fc_input_dim, 512), nn.ReLU(),
                LinearLayer(512, self.num_actions * self.num_atoms) # Output atoms even if not distributional
            )

        # Apply standard weight initialization only if not using Noisy Layers
        if not self.use_noisy:
             self.apply(init_weights)

    def _get_conv_out(self, shape):
        o = self.conv_layers(shape)
        return int(np.prod(o.size()))

    def forward(self, x):
        # Normalize input
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
                 # Reshape to [batch_size, num_actions/1, num_atoms]
                 value = value.view(batch_size, 1, self.num_atoms)
                 advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)
                 # Combine value and advantage streams
                 q_logits = value + advantage - advantage.mean(dim=1, keepdim=True)
                 q_dist = F.softmax(q_logits, dim=2) # Softmax over atoms
                 return q_dist, q_logits # Return distribution and logits
            else:
                 # Standard Dueling (num_atoms=1)
                 value = value.view(batch_size, 1)
                 advantage = advantage.view(batch_size, self.num_actions)
                 # Combine value and advantage streams
                 q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
                 return q_values # Return Q-values directly
        else:
            # Non-Dueling Architecture
            output = self.fc_layers(fc_input)
            if self.use_distributional:
                 q_logits = output.view(batch_size, self.num_actions, self.num_atoms)
                 q_dist = F.softmax(q_logits, dim=2)
                 return q_dist, q_logits # Return distribution and logits
            else:
                 # Standard DQN (num_atoms=1 implicitly)
                 q_values = output.view(batch_size, self.num_actions)
                 return q_values # Return Q-values directly

    def reset_noise(self):
        # Method to reset noise, typically called during training if needed.
        # Evaluation uses deterministic weights (mu) in the NoisyLinear forward pass.
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class AtariPreprocessor:
    # Copied from dqn2.py (same as test_model.py)
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        # Ensure input is handled correctly (RGB or Grayscale)
        if len(obs.shape) == 3 and obs.shape[2] == 3:
             gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif len(obs.shape) == 2:
             gray = obs # Already grayscale
        else:
            # Handle potential single-channel image (e.g., shape [H, W, 1])
            if len(obs.shape) == 3 and obs.shape[2] == 1:
                gray = obs.squeeze(-1) # Remove the last dimension
            else:
                raise ValueError(f"Unexpected observation shape: {obs.shape}")

        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        # Initialize the deque with the first frame replicated `frame_stack` times
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0) # Stack along the first axis (channel dimension)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        # Stack along the first axis (channel dimension)
        return np.stack(self.frames, axis=0)

# --- Evaluation Logic (Adapted from test_model.py) ---

def evaluate(cli_args): # Renamed internal arg to avoid conflict
    device = torch.device(cli_args.device)
    print(f"Using device: {device}")

    # --- Set Seed ---
    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cli_args.seed)
        torch.cuda.manual_seed_all(cli_args.seed) # if use multi-GPU

    # --- Environment Setup ---
    # Use render_mode="rgb_array" for recording frames
    env = gym.make(cli_args.env)
    # Seed the environment for reproducibility in evaluation
    env.action_space.seed(cli_args.seed)
    # Note: env.reset() needs seed argument for newer gym versions >= 0.26
    # env.observation_space.seed(cli_args.seed) # Deprecated in newer gym

    is_atari = "ALE/" in cli_args.env
    if is_atari:
        preprocessor = AtariPreprocessor()
        print("Using Atari Preprocessor.")
    else:
        # Handle non-Atari env state if needed (current DQN structure assumes image-like input)
        if cli_args.env != "ALE/Pong-v5":
             print(f"Warning: The DQN architecture is designed for Atari (4x84x84 input). Evaluating on {cli_args.env} might not work as expected.")
        preprocessor = None # No standard preprocessor for non-Atari

    num_actions = env.action_space.n

    # --- Create Model Configuration ---
    # Build an args namespace object mimicking dqn2.py's args for model init
    model_args = argparse.Namespace(
        use_dueling=cli_args.use_dueling,
        use_noisy=cli_args.use_noisy,
        use_distributional=cli_args.use_distributional,
        num_atoms=cli_args.num_atoms,
        v_min=cli_args.v_min,
        v_max=cli_args.v_max,
        device=device # Pass the device to the model args
    )
    print("\nInitializing model with configuration:")
    print(f"  Dueling: {model_args.use_dueling}")
    print(f"  Noisy: {model_args.use_noisy}")
    print(f"  Distributional: {model_args.use_distributional}")
    if model_args.use_distributional:
        print(f"    Num Atoms: {model_args.num_atoms}")
        print(f"    V_min: {model_args.v_min}")
        print(f"    V_max: {model_args.v_max}")

    # --- Initialize Model ---
    model = DQN(num_actions, model_args).to(device)

    # --- Load Model Weights ---
    try:
        # dqn2.py saves the state_dict directly
        state_dict = torch.load(cli_args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"\nModel loaded successfully from {cli_args.model_path}")
    except Exception as e:
        print(f"Error loading model state_dict from {cli_args.model_path}: {e}")
        env.close()
        return

    # --- Set Model to Evaluation Mode ---
    # This is crucial:
    # - Disables dropout
    # - Affects batch normalization layers
    # - Switches NoisyLinear layers to use deterministic weights (mu)
    model.eval()
    print("Model set to evaluation mode.")

    # --- Create Output Directory ---
    os.makedirs(cli_args.output_dir, exist_ok=True)

    # --- Run Evaluation Episodes ---
    total_reward_list = []
    max_eval_steps = 27000 # Set a max step limit consistent with training?

    for ep in range(cli_args.episodes):
        # Use seed for reset for reproducibility per episode
        obs, info = env.reset(seed=cli_args.seed + ep)

        if is_atari:
            state = preprocessor.reset(obs)
        else:
            state = obs # Assume non-Atari state is ready

        done = False
        truncated = False # Handle truncated episodes
        total_reward = 0
        frames = []
        step_count = 0

        print(f"\nStarting evaluation episode {ep+1}/{cli_args.episodes}...")

        while not done and not truncated and step_count < max_eval_steps:
            # Render frame BEFORE taking a step
            frame = env.render()
            frames.append(frame)

            # Prepare state tensor
            # Ensure state is float32 numpy array before converting
            state_np = np.array(state, dtype=np.float32)
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(device)

            # Select action greedily based on Q-values
            with torch.no_grad(): # Ensure no gradients are computed
                output = model(state_tensor)

                # Handle output based on whether the model is distributional
                if model_args.use_distributional:
                    q_dist, _ = output # Get the distribution [1, num_actions, num_atoms]
                    # Calculate expected Q-values: sum(probability * support_value)
                    q_values = (q_dist * model.support.unsqueeze(0).unsqueeze(0)).sum(2) # Shape: [1, num_actions]
                else:
                    q_values = output # Output is already Q-values [1, num_actions]

                action = q_values.argmax().item() # Choose action with highest Q-value

            # Take action in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated # Episode ends if terminated

            total_reward += reward
            step_count += 1

            # Update state
            if is_atari:
                state = preprocessor.step(next_obs)
            else:
                state = next_obs # Update for next iteration

            if done or truncated:
                print(f"Episode {ep+1} finished after {step_count} steps. Terminated: {terminated}, Truncated: {truncated}")


        # --- Save Episode Video ---
        out_path = os.path.join(cli_args.output_dir, f"eval_ep{ep+1}_{cli_args.env.split('/')[-1]}.mp4")
        try:
            # Use imageio to save the recorded frames as MP4
            with imageio.get_writer(out_path, fps=30) as video:
                for f in frames:
                    video.append_data(f)
            print(f"Saved episode {ep+1} video with total reward {total_reward:.2f} -> {out_path}")
        except Exception as e:
            print(f"Error saving video for episode {ep+1}: {e}")

        total_reward_list.append(total_reward)

    # --- Cleanup ---
    env.close()

    # --- Print Average Reward ---
    if total_reward_list:
        avg_reward = np.mean(total_reward_list)
        std_reward = np.std(total_reward_list)
        print(f"\n--- Evaluation Summary ---")
        print(f"Average reward over {cli_args.episodes} episodes: {avg_reward:.2f} +/- {std_reward:.2f}")
        print(f"Individual rewards: {total_reward_list}")
    else:
        print("\nNo complete episodes were evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Rainbow DQN agent trained with dqn2.py")

    # --- Evaluation Parameters (from test_model.py) ---
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained .pt model state_dict file")
    parser.add_argument("--output-dir", type=str, default="./eval_videos_rainbow", help="Directory to save evaluation videos")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for evaluation reproducibility")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", choices=["ALE/Pong-v5"], help="Gym environment name (currently only Atari supported well by model)") # Limited choices for now

    # --- Hardware Parameters ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation (cuda or cpu)")

    # --- Model Configuration Parameters (MUST match the trained model's config) ---
    # These flags tell the script how to initialize the DQN architecture
    parser.add_argument("--use-dueling", action='store_true', help="Specify if the loaded model uses Dueling architecture")
    parser.add_argument("--use-noisy", action='store_true', help="Specify if the loaded model uses Noisy Nets")
    parser.add_argument("--use-distributional", action='store_true', help="Specify if the loaded model uses Distributional RL (C51)")

    # --- Distributional RL Specific Parameters (Required if --use-distributional) ---
    parser.add_argument("--num-atoms", type=int, default=51, help="Number of atoms used in Distributional RL (must match training)")
    parser.add_argument("--v-min", type=float, default=-10.0, help="Minimum value of the support range (must match training)")
    parser.add_argument("--v-max", type=float, default=10.0, help="Maximum value of the support range (must match training)")

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.use_distributional:
        print("Distributional RL enabled. Ensure --num-atoms, --v-min, and --v-max match the trained model.")
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        exit(1)

    evaluate(args)