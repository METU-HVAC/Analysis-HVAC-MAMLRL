import sinergym
import gym
import torch
import wandb
import argparse
from distutils.util import strtobool
import torch.nn.functional as F
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import copy
from tools.modelHelpers import * 
# class ReplayBuffer:
#     def __init__(self, obs_dim, buffer_size, batch_size, device):
#         self.obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
#         self.next_obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
#         self.acts_buf = np.zeros(buffer_size, dtype=np.int64)
#         self.rews_buf = np.zeros(buffer_size, dtype=np.float32)
#         self.done_buf = np.zeros(buffer_size, dtype=np.float32)
#         self.max_size = buffer_size
#         self.batch_size = batch_size
#         self.ptr = 0
#         self.size = 0
#         self.device = device

#     def store(self, obs, act, rew, next_obs, done):
#         self.obs_buf[self.ptr] = obs
#         self.next_obs_buf[self.ptr] = next_obs
#         self.acts_buf[self.ptr] = act
#         self.rews_buf[self.ptr] = rew
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)

#     def sample_batch(self):
#         idxs = np.random.choice(self.size, self.batch_size, replace=False)
#         batch = dict(obs=self.obs_buf[idxs],
#                      next_obs=self.next_obs_buf[idxs],
#                      acts=self.acts_buf[idxs],
#                      rews=self.rews_buf[idxs],
#                      done=self.done_buf[idxs])
#         return {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
    

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        # Q2 architecture
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=1.0)
        torch.nn.init.constant_(m.bias, 0)

class SACDiscrete(nn.Module):
    def __init__(self, obs_dim, action_dim, updates_per_step, device, learning_rate, buffer_size, batch_size):
        super(SACDiscrete, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        
        self.updates_per_step = updates_per_step
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # Automatic entropy tuning
        self.target_entropy = -np.log(1.0 / action_dim) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        self.gamma=0.99
        self.tau=0.005
        self.replay_buffer = ReplayBuffer(obs_dim, self.buffer_size, self.batch_size, self.device)

    def choose_action(self, state, greedy=False):
        logits = self.actor(state)
        if greedy:
            probs = F.softmax(logits, dim=-1)

            # Create a Categorical distribution over the action space
            dist = Categorical(probs=probs)

            # Sample one action
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=1)       

        return action
    
    def load_model(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        # Load model parameters
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.critic_target.apply(init_weights)
        self.replay_buffer.deserialize(checkpoint["replay_buffer"])  # Load replay buffer
        
        # Load optimizers
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        
        # Load alpha value
        self.log_alpha = torch.tensor(checkpoint["log_alpha"], requires_grad=True, device=self.device)
        
        # Reload hyperparameters if needed (optional)
        hyperparams = checkpoint["hyperparameters"]
        self.gamma = hyperparams["gamma"]
        self.tau = hyperparams["tau"]
        self.target_entropy = hyperparams["target_entropy"]
        
        print(f"Model loaded from {load_path}")

    def save_model(self, save_path):
        checkpoint = {
        "actor_state_dict": self.actor.state_dict(),
        "critic_state_dict": self.critic.state_dict(),
        "critic_target_state_dict": self.critic_target.state_dict(),
        "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        "log_alpha": self.log_alpha.detach().cpu().numpy(),  # Save as a numpy array
        "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
        "replay_buffer": self.replay_buffer.serialize(),  # Add replay buffer
        "hyperparameters": {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
            "updates_per_step": self.updates_per_step,
        },
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved at {save_path}")

    def update(self):
        if self.replay_buffer.size >= self.batch_size:
            for _ in range(self.updates_per_step):
                batch = self.replay_buffer.sample_batch()
    
                with torch.no_grad():
                    next_state_action_logits = self.actor(batch['next_obs'])
                    next_state_probs = F.softmax(next_state_action_logits, dim=-1)
                    next_state_log_probs = F.log_softmax(next_state_action_logits, dim=-1)
                    
                    next_state_m = Categorical(probs=next_state_probs)
                    next_state_actions = next_state_m.sample()
                    next_q1_target, next_q2_target = self.critic_target(batch['next_obs'])
                    next_q_target = torch.min(next_q1_target, next_q2_target)
                    next_q = (next_state_probs * (next_q_target - self.log_alpha.exp() * next_state_log_probs)).sum(dim=-1)

                    td_target = batch['rews'] + (1 - batch['done']) * self.gamma * next_q

                # Critic loss
                q1, q2 = self.critic(batch['obs'])
                q1_pred = q1.gather(1, batch['acts'].unsqueeze(-1)).squeeze(-1)
                q2_pred = q2.gather(1, batch['acts'].unsqueeze(-1)).squeeze(-1)
                critic_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

                # Actor loss
                state_action_logits = self.actor(batch['obs'])
                state_probs = F.softmax(state_action_logits, dim=-1)
                state_log_probs = F.log_softmax(state_action_logits, dim=-1)
                with torch.no_grad():
                    q1_pi, q2_pi = self.critic(batch['obs'])
                    min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (state_probs * (self.log_alpha.exp() * state_log_probs - min_q_pi)).sum(dim=1).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                # Entropy temperature adjustment
                alpha_loss = -(self.log_alpha * (state_log_probs.detach() + self.target_entropy).mean())
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward(retain_graph=True)
                self.alpha_optimizer.step()

                # Soft update of target network
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return actor_loss, critic_loss