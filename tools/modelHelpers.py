import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class SequentialReplayBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, act_shape, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        # Initialize buffers for storing temporal data
        self.obs_buf = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.next_obs_buf = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.acts_buf = torch.zeros((num_steps, num_envs), device=device)
        self.logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
        self.rews_buf = torch.zeros((num_steps, num_envs), device=device)
        self.dones_buf = torch.zeros((num_steps, num_envs), device=device)
        self.values_buf = torch.zeros((num_steps, num_envs), device=device)
        
        self.ptr = 0  # Pointer to the current step

    def store(self, obs, act, logprob, rew, next_obs, done, value):
        """Store a single timestep of data for all environments."""
        self.obs_buf[self.ptr] = obs.clone().detach()
        self.next_obs_buf[self.ptr] = next_obs.clone().detach()
        self.acts_buf[self.ptr] = act.clone().detach()
        self.logprobs_buf[self.ptr] = logprob.clone().detach()
        self.rews_buf[self.ptr] = rew.clone().detach()
        self.dones_buf[self.ptr] = done.clone().detach()
        self.values_buf[self.ptr] = value.clone().detach()

        # Increment pointer
        self.ptr = (self.ptr + 1) % self.num_steps

    def get_batch(self):
        """Retrieve the full batch of data as a dictionary."""
        batch = {
            'obs': self.obs_buf,
            'next_obs': self.next_obs_buf,
            'acts': self.acts_buf,
            'logprobs': self.logprobs_buf,
            'rews': self.rews_buf,
            'dones': self.dones_buf,
            'values': self.values_buf
        }
        # Convert to tensors on the specified device
        return {k: v.clone().to(self.device) for k, v in batch.items()}

    def reset(self):
        """Reset the buffer and pointer for the next trajectory."""
        self.ptr = 0
        self.obs_buf.fill_(0)
        self.next_obs_buf.fill_(0)
        self.acts_buf.fill_(0)
        self.logprobs_buf.fill_(0)
        self.rews_buf.fill_(0)
        self.dones_buf.fill_(0)
        self.values_buf.fill_(0)

    def __len__(self):
        return self.num_steps
    
    def get_state(self):
        """Return the state of the replay buffer as a dictionary."""
        return {
            "obs_buf": self.obs_buf,
            "next_obs_buf": self.next_obs_buf,
            "acts_buf": self.acts_buf,
            "rews_buf": self.rews_buf,
            "logprobs_buf": self.logprobs_buf,
            "dones_buf": self.dones_buf,
            "values_buf": self.values_buf,
            "ptr": self.ptr
        }

    def set_state(self, state):
        """Set the replay buffer state from a dictionary."""
        self.obs_buf = state["obs_buf"]
        self.next_obs_buf = state["next_obs_buf"]
        self.acts_buf = state["acts_buf"]
        self.rews_buf = state["rews_buf"]
        self.dones_buf = state["dones_buf"]
        self.logprobs_buf = state["logprobs_buf"]
        self.values_buf = state["values_buf"]
        self.ptr = state["ptr"]

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, obs_dim, buffer_size, batch_size, device):
        self.obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(buffer_size, dtype=np.int64)
        self.rews_buf = np.zeros(buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)
        self.max_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.choice(self.size, self.batch_size, replace=False)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     acts=self.acts_buf[idxs],
                     rews=self.rews_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
    
    def __len__(self):
        return len(self.buffer)
    
    def serialize(self):
        """
        Converts the replay buffer into a dictionary for saving.
        """
        return {
            "obs_buf": self.obs_buf,
            "next_obs_buf": self.next_obs_buf,
            "acts_buf": self.acts_buf,
            "rews_buf": self.rews_buf,
            "done_buf": self.done_buf,
            "max_size": self.max_size,
            "batch_size": self.batch_size,
            "ptr": self.ptr,
            "size": self.size
        }

    def deserialize(self, data):
        """
        Loads the replay buffer from a dictionary.
        """
        self.obs_buf = data["obs_buf"]
        self.next_obs_buf = data["next_obs_buf"]
        self.acts_buf = data["acts_buf"]
        self.rews_buf = data["rews_buf"]
        self.done_buf = data["done_buf"]
        self.max_size = data["max_size"]
        self.batch_size = data["batch_size"]
        self.ptr = data["ptr"]
        self.size = data["size"]

# Define the Policy Model
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.output_layer(x)
        return x
