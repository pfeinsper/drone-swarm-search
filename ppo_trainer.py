import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import json
import os
from datetime import datetime

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches

class ActorCriticNetwork(nn.Module):
    def __init__(self, grid_size, n_actions):
        super().__init__()
        
        # CNN for processing probability matrix
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling to fixed size
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, grid_size, grid_size)
            cnn_out_size = self.cnn(dummy_input).shape[1]
        
        # Process position and battery info
        self.state_net = nn.Sequential(
            nn.Linear(4, 64),  # position(2) + battery(1) + distance_to_base(1)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combine CNN and state features
        self.shared_net = nn.Sequential(
            nn.Linear(cnn_out_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor (policy) head with separate advantages for each action
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        # Process probability matrix through CNN
        prob_matrix = state['probability_matrix'].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        cnn_features = self.cnn(prob_matrix)
        
        # Process other features
        position = state['position'].unsqueeze(0)  # [1, 2]
        battery = state['battery_level'].unsqueeze(0)  # [1, 1]
        distance = torch.norm(state['distance_to_base']).unsqueeze(0).unsqueeze(0)  # [1, 1]
        
        # Combine position and battery features
        state_features = torch.cat([position, battery, distance], dim=1)  # [1, 4]
        state_features = self.state_net(state_features)
        
        # Combine all features
        combined_features = torch.cat([cnn_features, state_features], dim=1)
        shared_features = self.shared_net(combined_features)
        
        # Get action probabilities and state value
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        
        return action_probs.squeeze(0), state_value.squeeze(0)

class PPOTrainer:
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=64,
        n_epochs=10,
        log_dir='logs',
        initial_entropy_coef=0.1,  # Initial entropy coefficient for exploration
        final_entropy_coef=0.01,   # Final entropy coefficient
        exploration_fraction=0.3,   # Fraction of training where we anneal exploration
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.initial_entropy_coef = initial_entropy_coef
        self.final_entropy_coef = final_entropy_coef
        self.entropy_coef = initial_entropy_coef
        self.exploration_fraction = exploration_fraction
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Initialize network and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(
            grid_size=env.grid_size,
            n_actions=len(env.possible_actions)
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize memory buffer
        self.memory = PPOMemory(batch_size)
        
        # Setup logging
        self.log_dir = log_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.energy_consumed = []
        self.recharge_counts = []
        self.targets_found = []
        
        # Save hyperparameters
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon": epsilon,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "grid_size": env.grid_size,
            "drone_amount": env.drone.amount,
            "timestep_limit": env.timestep_limit
        })
    
    def save_hyperparameters(self, hyperparams):
        """Save hyperparameters to a JSON file"""
        with open(os.path.join(self.run_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)
    
    def log_metrics(self, metrics, episode):
        """Log metrics for the current episode"""
        for key, value in metrics.items():
            if key == "episode_reward":
                self.episode_rewards.append(value)
            elif key == "episode_length":
                self.episode_lengths.append(value)
            elif key == "energy_consumed":
                self.energy_consumed.append(value)
            elif key == "recharge_count":
                self.recharge_counts.append(value)
            elif key == "targets_found":
                self.targets_found.append(value)
    
    def plot_metrics(self):
        """Plot and save training metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # Plot energy metrics
        plt.subplot(2, 2, 3)
        plt.plot(self.energy_consumed, label='Energy Consumed')
        plt.plot(self.recharge_counts, label='Recharge Count')
        plt.title('Energy Metrics')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot targets found
        plt.subplot(2, 2, 4)
        plt.plot(self.targets_found)
        plt.title('Targets Found')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'training_metrics.png'))
        plt.close()
        
        # Save metrics to CSV
        np.savetxt(os.path.join(self.run_dir, 'metrics.csv'), 
                  np.column_stack([
                      self.episode_rewards,
                      self.episode_lengths,
                      self.energy_consumed,
                      self.recharge_counts,
                      self.targets_found
                  ]),
                  delimiter=',',
                  header='rewards,lengths,energy,recharges,targets',
                  comments='')
    
    def preprocess_state(self, state):
        """Convert numpy arrays to PyTorch tensors and move to device"""
        return {
            'position': torch.FloatTensor(state['position']).to(self.device),
            'probability_matrix': torch.FloatTensor(state['probability_matrix']).to(self.device),
            'battery_level': torch.FloatTensor(state['battery_level']).to(self.device),
            'distance_to_base': torch.FloatTensor(state['distance_to_base']).to(self.device)
        }
    
    def choose_action(self, state):
        """Select action using the current policy with added exploration"""
        with torch.no_grad():
            state = self.preprocess_state(state)
            action_probs, state_value = self.network(state)
            
            # Add exploration noise to probabilities
            action_probs = action_probs + torch.randn_like(action_probs) * self.entropy_coef
            action_probs = torch.softmax(action_probs, dim=-1)  # Renormalize
            
            # Ensure no probability is too close to 0 or 1
            action_probs = torch.clamp(action_probs, min=0.01, max=0.99)
            action_probs = action_probs / action_probs.sum()  # Renormalize again
            
            dist = Categorical(action_probs)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action).item(), state_value.item()
    
    def update_exploration(self, progress):
        """Update entropy coefficient based on training progress"""
        if progress <= self.exploration_fraction:
            # Linearly anneal from initial to final value
            alpha = 1.0 - (progress / self.exploration_fraction)
            self.entropy_coef = self.final_entropy_coef + (self.initial_entropy_coef - self.final_entropy_coef) * alpha
        else:
            self.entropy_coef = self.final_entropy_coef

    def save_model(self, filename='model.pt'):
        """Save the current model state"""
        path = os.path.join(self.run_dir, filename)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'energy_consumed': self.energy_consumed,
            'recharge_counts': self.recharge_counts,
            'targets_found': self.targets_found,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a saved model state"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.energy_consumed = checkpoint.get('energy_consumed', [])
        self.recharge_counts = checkpoint.get('recharge_counts', [])
        self.targets_found = checkpoint.get('targets_found', [])
        print(f"Model loaded from {path}")

    def train(self, n_episodes):
        """Main training loop with improved exploration and model saving"""
        print(f"Starting training... Logs will be saved to {self.run_dir}")
        
        best_reward = float('-inf')
        no_improvement_count = 0
        
        for episode in range(n_episodes):
            # Update exploration
            progress = episode / n_episodes
            self.update_exploration(progress)
            
            episode_reward = 0
            episode_steps = 0
            
            # Reset environment and memory
            state, _ = self.env.reset()  # Drones will start with full battery
            self.memory.clear()
            done = False
            
            # Collect trajectory
            while not done and episode_steps < self.env.timestep_limit:
                # Get action for each drone
                current_actions = {}
                
                for agent in self.env.agents:
                    action, log_prob, value = self.choose_action(state[agent])
                    current_actions[agent] = action
                    
                    # Store experience in memory
                    self.memory.store(
                        state[agent],
                        action,
                        log_prob,
                        value,
                        0.0,  # placeholder reward
                        False  # placeholder done
                    )
                
                # Take action in environment
                next_state, reward, termination, truncation, _ = self.env.step(current_actions)
                
                # Update rewards and done flags in memory
                for idx, agent in enumerate(self.env.agents):
                    if agent in reward:
                        self.memory.rewards[-len(self.env.agents) + idx] = reward[agent]
                        is_done = termination.get(agent, False) or truncation.get(agent, False)
                        self.memory.dones[-len(self.env.agents) + idx] = is_done
                        episode_reward += reward[agent]
                
                # Update state and check if done
                state = next_state
                done = any(termination.values()) or any(truncation.values())
                episode_steps += 1
            
            # Handle early termination
            if done and episode_steps < self.env.timestep_limit:
                # Add small penalty for early termination due to battery depletion
                for idx in range(len(self.memory.rewards) - len(self.env.agents), len(self.memory.rewards)):
                    if self.memory.dones[idx]:
                        self.memory.rewards[idx] -= 1.0
            
            # Calculate returns and advantages
            returns = self.compute_returns(self.memory.rewards)
            advantages = self.compute_advantages(returns, self.memory.vals)
            
            # Generate training batches
            batches = self.memory.generate_batches()
            
            # Update policy for each batch
            for batch_indices in batches:
                states = [self.memory.states[i] for i in batch_indices]
                actions = [self.memory.actions[i] for i in batch_indices]
                old_log_probs = [self.memory.probs[i] for i in batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                self.update_policy(states, actions, old_log_probs, batch_returns, batch_advantages)
            
            # Log metrics
            metrics = {
                "episode_reward": episode_reward,
                "episode_length": episode_steps,
                **self.env.episode_metrics
            }
            self.log_metrics(metrics, episode)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_model('best_model.pt')
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Save periodic checkpoints
            if episode % 100 == 0:
                self.save_model(f'checkpoint_episode_{episode}.pt')
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}, "
                      f"Entropy = {self.entropy_coef:.3f}, Best = {best_reward:.2f}")
                self.plot_metrics()
        
        # Save final model
        self.save_model('final_model.pt')
        
        # Final metrics plot
        self.plot_metrics()
        print(f"\nTraining completed! Results saved to {self.run_dir}")
        print(f"Average reward over last 100 episodes: {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"Success rate: {np.mean(self.targets_found) * 100:.2f}%")
        print(f"Best reward achieved: {best_reward:.2f}")
    
    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.FloatTensor(returns).to(self.device)
    
    def compute_advantages(self, returns, values):
        """Compute advantages"""
        advantages = returns - torch.FloatTensor(values).to(self.device)
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def update_policy(self, states, actions, old_log_probs, returns, advantages):
        """Update policy using PPO"""
        # Convert to tensors
        states = [self.preprocess_state(s) for s in states]
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Create dataset
        dataset = list(zip(states, actions, old_log_probs, returns, advantages))
        
        # Update for n_epochs
        for _ in range(self.n_epochs):
            # Sample random mini-batches
            np.random.shuffle(dataset)
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i + self.batch_size]
                batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = zip(*batch)
                
                # Convert batch data to tensors
                batch_actions = torch.stack([a if torch.is_tensor(a) else torch.tensor(a) for a in batch_actions]).to(self.device)
                batch_old_log_probs = torch.stack([lp if torch.is_tensor(lp) else torch.tensor(lp) for lp in batch_old_log_probs]).to(self.device)
                batch_returns = torch.stack([r if torch.is_tensor(r) else torch.tensor(r) for r in batch_returns]).to(self.device)
                batch_advantages = torch.stack([adv if torch.is_tensor(adv) else torch.tensor(adv) for adv in batch_advantages]).to(self.device)
                
                # Forward pass
                action_probs, values = zip(*[self.network(s) for s in batch_states])
                action_probs = torch.stack(action_probs)
                values = torch.cat(values)
                
                # Calculate new log probs
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate ratio and clipped ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                
                # Calculate losses
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                value_loss = 0.5 * (values - batch_returns).pow(2).mean()
                
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

# Default hyperparameters
hyperparameters = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "batch_size": 64,
    "n_epochs": 10,
    "log_dir": "logs"  # Changed from use_wandb to log_dir
}

if __name__ == "__main__":
    # This section only runs if ppo_trainer.py is run directly
    from basic_env2 import EnergyAwareDroneSwarmSearch
    
    # Create environment
    env = EnergyAwareDroneSwarmSearch(
        grid_size=40,
        render_mode="human",
        render_grid=True,
        render_gradient=True,
        vector=(1, 1),
        timestep_limit=300,
        person_amount=1,
        dispersion_inc=0.05,
        person_initial_position=(15, 15),
        drone_amount=4,
        drone_speed=10,
        probability_of_detection=0.9,
        pre_render_time=0,
    )
    
    # Create trainer and start training
    trainer = PPOTrainer(env, **hyperparameters)
    trainer.train(n_episodes=1000) 