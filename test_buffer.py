import numpy as np
import matplotlib.pyplot as plt
from dqn import PrioritizedReplayBuffer

def test_prioritized_replay_buffer():
    # Initialize buffer
    buffer = PrioritizedReplayBuffer(1000)
    print("Adding data to buffer...")

    # Generate test data
    states = np.random.rand(1000, 4)
    actions = np.random.randint(0, 2, 1000)
    rewards = np.random.rand(1000)
    next_states = np.random.rand(1000, 4)
    dones = np.random.randint(0, 2, 1000)

    # Add samples to buffer with random TD errors as priorities
    for i in range(1000):
        transition = (states[i], actions[i], rewards[i], next_states[i], dones[i])
        td_error = np.random.rand()  # Random TD error
        buffer.add(transition, td_error)

    # Test sampling
    print("\nTesting sampling...")
    batch, indices, weights = buffer.sample(32)
    print(f"Sample size: {len(batch)}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights range: [{weights.min():.1f}, {weights.max():.1f}]")

    # Plot weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=20, alpha=0.7)
    plt.title('Weight Distribution - Single Batch')
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('weights_distribution.png')
    plt.savefig('weights_distribution.eps', format='eps')
    plt.close()

    # Test priority updates
    print("\nTesting priority updates...")
    
    # Get weights before update
    weights_before = weights
    
    # Update priorities with new random values
    new_priorities = np.random.rand(len(indices))
    for idx, priority in zip(indices, new_priorities):
        buffer.add(buffer.buffer[idx], priority)
    
    # Get weights after update
    batch_after, _, weights_after = buffer.sample(32)
    
    # Plot weight comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(weights_before, bins=20, alpha=0.7)
    plt.title('Weights Before Update')
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(weights_after, bins=20, alpha=0.7)
    plt.title('Weights After Update')
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('weights_comparison.png')
    plt.savefig('weights_comparison.eps', format='eps')
    plt.close()

    # Test buffer when full
    print("\nTesting buffer when full...")
    print(f"Current buffer size: {len(buffer.buffer)}")
    
    # Add more samples to test overflow behavior
    for i in range(100):
        transition = (states[i], actions[i], rewards[i], next_states[i], dones[i])
        buffer.add(transition, np.random.rand())
    print(f"Buffer size after adding more samples: {len(buffer.buffer)}")
    
    # Test multiple sampling
    print("\nTesting multiple sampling...")
    plt.figure(figsize=(12, 6))
    for i in range(3):
        _, _, batch_weights = buffer.sample(32)
        plt.subplot(1, 3, i+1)
        plt.hist(batch_weights, bins=20, alpha=0.7)
        plt.title(f'Batch {i+1} Weights')
        plt.xlabel('Weight Values')
        plt.ylabel('Frequency')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('multiple_batches.png')
    plt.savefig('multiple_batches.eps', format='eps')
    plt.close()

    print("\nTests completed. Please check the generated charts:")
    print("1. weights_distribution.png/eps - Shows weight distribution for a single batch")
    print("2. weights_comparison.png/eps - Shows weight distribution before and after priority updates")
    print("3. multiple_batches.png/eps - Shows weight distributions for multiple sampling batches")

if __name__ == "__main__":
    test_prioritized_replay_buffer() 