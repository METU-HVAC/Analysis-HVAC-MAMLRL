import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
from tools.modelHelpers import Policy, ReplayBuffer

# Define the Agent
class DDQNAgent(nn.Module):
    def __init__(self, num_states, num_actions, gamma=0.99, lr=0.001, buffer_capacity=100000, batch_size=64, device='cpu'):
        super(DDQNAgent, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = device

        # Initialize policy and target models
        self.policy = Policy(num_states, num_actions).to(device)
        self.model_target = Policy(num_states, num_actions).to(device)
        self.model_target.load_state_dict(self.policy.state_dict())  # Copy initial weights
        self.model_target.eval()  # Target model is not trained

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(num_states, buffer_capacity, batch_size, device)
        self.batch_size = batch_size

    def update(self):
        """Perform a training step."""
        if self.replay_buffer.size < self.batch_size:
            print("Not enough samples in the replay buffer!")
            return 0
        

        # Sample minibatch from replay buffer
        batch = self.replay_buffer.sample_batch()

        # Predict future rewards using the target model
        with torch.no_grad():
            future_rewards = self.model_target(batch['next_obs'])
            
            max_future_rewards = future_rewards.max(dim=1)[0]  # Take max Q-value for each next state
            updated_q_values = batch['rews'] + self.gamma * max_future_rewards * (1 - batch['done'])
        
        actions = batch['acts']
        assert (actions >= 0).all() and (actions < self.num_actions).all(), \
        f"Invalid action found in batch: {actions}"
        # One-hot encode actions
        masks = F.one_hot(batch['acts'], num_classes=self.num_actions).float()

        # Compute the loss
        q_values = self.policy(batch['obs'])  # Forward pass through policy model
        q_action = (q_values * masks).sum(dim=1)  # Get Q-values for taken actions
        
        loss = self.loss_function(q_action, updated_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss

    def choose_action(self, state, epsilon):
        """Choose an action using epsilon-greedy policy."""
        if random.random() < epsilon:
            # Explore: choose a random action
            return random.randint(0, self.num_actions - 1)
        else:
            # # Exploit: choose the best action based on Q-values
            # state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # with torch.no_grad():
            #     q_values = self.policy(state_tensor)
            # return torch.argmax(q_values, dim=1).item()
            return self.choose_greedy_action(state)
    def choose_greedy_action(self, state):
        # Exploit: choose the best action based on Q-values
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy(state_tensor)
        return torch.argmax(q_values, dim=1).item()
    
    # Save the state dict
    def save_ddqn_agent(self, filepath):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "model_target_state_dict": self.model_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_buffer": self.replay_buffer,  # Optional: save replay buffer if needed
            "gamma": self.gamma,
            "batch_size": self.batch_size,
        }, filepath)
        
    # Load the state dict
    def load_ddqn_agent(self, filepath, device="cpu"):
        checkpoint = torch.load(filepath, map_location=device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.model_target.load_state_dict(checkpoint["model_target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore hyperparameters and optionally replay buffer
        self.gamma = checkpoint["gamma"]
        self.batch_size = checkpoint["batch_size"]
        if "replay_buffer" in checkpoint:
            self.replay_buffer = checkpoint["replay_buffer"]
        
        print(f"DDQNAgent state loaded from {filepath}")

