import sinergym
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sinergym.utils.wrappers import (NormalizeAction, NormalizeObservation, ReduceObservationWrapper)
import wandb
from environment import *
import random
import argparse
from distutils.util import strtobool
import torch.nn.functional as F
import os
import time
import numpy as np
from collections import deque
import copy

from agents.models.SACAgent import SACDiscrete

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


def make_env(gym_id, seed, energy_weight, lambda_temperature, obs_to_remove):
    def thunk():
        env = gym.make(
            gym_id,
            env_name="env" + str(seed),
            reward=MyCustomReward,
            reward_kwargs={
                "temperature_variables": ["air_temperature"],
                "energy_variables": ["HVAC_electricity_demand_rate"],
                "range_comfort_winter": [20.0, 23.5],
                "range_comfort_summer": [23.0, 26.0],
                "summer_start": [6, 1],
                "summer_final": [9, 30],
                "energy_weight": energy_weight,
                "lambda_energy": 1e-4,
                "lambda_temperature": 1,
            },
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NormalizeObservation(env)
        env = ReduceObservationWrapper(env=env, obs_reduction=obs_to_remove)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        # Q2 architecture
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2
    
def save_model(obs_dim, action_dim, learning_rate, replay_buffer, buffer_size, batch_size, gamma, tau, target_entropy, updates_per_step, actor, critic, critic_target, actor_optimizer, critic_optimizer, log_alpha, alpha_optimizer, save_path):
    checkpoint = {
    "actor_state_dict": actor.state_dict(),
    "critic_state_dict": critic.state_dict(),
    "critic_target_state_dict": critic_target.state_dict(),
    "actor_optimizer_state_dict": actor_optimizer.state_dict(),
    "critic_optimizer_state_dict": critic_optimizer.state_dict(),
    "log_alpha": log_alpha.detach().cpu().numpy(),  # Save as a numpy array
    "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
    "replay_buffer": replay_buffer.serialize(),  # Add replay buffer
    "hyperparameters": {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "target_entropy": target_entropy,
        "updates_per_step": updates_per_step,
    },
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved at {save_path}")
    
def train(config=None):
    with wandb.init(config=config, reinit=True, settings=wandb.Settings(start_method="thread")):
        config = wandb.config

        device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
        # Observation dimensions
        all_obs_list = ['month', 'day_of_month', 'hour', 'outdoor_temperature', 'outdoor_humidity',
                        'wind_speed', 'wind_direction', 'diffuse_solar_radiation', 'direct_solar_radiation',
                        'htg_setpoint', 'clg_setpoint', 'air_temperature', 'air_humidity', 'people_occupant',
                        'co2_emission', 'HVAC_electricity_demand_rate', 'total_electricity_HVAC']
        obs_remove = []
        OBS_VARIABLES = [obs for obs in all_obs_list if obs not in obs_remove]
        print(OBS_VARIABLES)
        OBSERVATION_DIM = len(OBS_VARIABLES)
        max_ep_len = 52560
        num_epochs = 2
        num_steps = max_ep_len
        env_list = [config.climate_type]
        num_tasks = len(env_list)
        
        envs_list = []
        k = 0
        for i in env_list:
            k += 1
            env_name = f"Eplus-A403-{i}-discrete-stochastic-v1" 
            print(env_name)
            envs_list.append(make_env(
            gym_id=env_name, seed=42 + k,
            energy_weight=config.energy_weight,lambda_temperature=1,obs_to_remove=[]) 
            )
        envs = gym.vector.AsyncVectorEnv(envs_list)

        #Create an evaluation environment
        # eval_env = gym.vector.SyncVectorEnv([make_env(gym_id=environment, seed=42,energy_weight=config.energy_weight,lambda_temperature=0.01,obs_to_remove=[])])
        
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = 70
        print("Observation Dim: ", obs_dim, "Action Dim: ", action_dim)
        model = SACDiscrete(obs_dim=np.array(envs.single_observation_space.shape).prod(), action_dim=action_dim, updates_per_step=4, buffer_size=1000000, learning_rate=config.learning_rate, batch_size=config.batch_size, device=device).to(device)

        # actor = Actor(obs_dim, action_dim).to(device)
        # critic = Critic(obs_dim, action_dim).to(device)
        # critic_target = copy.deepcopy(critic).to(device)
        # critic_target.eval()

        # actor_optimizer = optim.Adam(actor.parameters(), lr=config.learning_rate)
        # critic_optimizer = optim.Adam(critic.parameters(), lr=config.learning_rate)

        # # Automatic entropy tuning
        # target_entropy = -np.log(1.0 / action_dim) * 0.98
        # log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # alpha_optimizer = optim.Adam([log_alpha], lr=config.learning_rate)

        # replay_buffer = ReplayBuffer(obs_dim, config.buffer_size, config.batch_size, device)

        rewards_list = [0.0] * num_tasks
        power_demand_list = [0.0] * num_tasks
        temp_violation_list = [0.0] * num_tasks

        start_time = time.time()
        for epoch in range(config.num_episodes):
            
            train_sac(config, envs, num_steps, model, device, rewards_list, power_demand_list, temp_violation_list, num_tasks, args.energy_weight, args.climate_type)
        
        wandb.log({"Execution Time": abs(start_time - time.time())})
        envs.close()

def train_sac(config, envs, num_steps, model,
             device, rewards_list, power_demand_list, temp_violation_list, num_tasks, energy_weight, climate_type, gamma=0.99, tau=0.005):
    global_step = 0
  
    state, _ = envs.reset()
    initial_steps = 20000
  
    for step in range(0, num_steps - 1):
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = model.choose_action(state_tensor)

        next_state, reward, done, truncated, info = envs.step(action.cpu().numpy())


        # if 'total_power_demand' in info and 'total_temperature_violation' in info:
        #     total_power_demand = info["total_power_demand"]
        #     total_temperature_violation = info["total_temperature_violation"]
        #     temp_violation_list += total_temperature_violation
        #     power_demand_list += total_power_demand
        #     # Append episodic return to the list
        #     rewards_list += reward
        # # Log the episodic return
        # wandb.log({"episodic_return": np.mean(rewards_list) / (global_step + 1)})
        # wandb.log({"avg_power":np.mean(power_demand_list)/ (global_step + 1)})
        # wandb.log({"temperature_violation":np.mean(temp_violation_list)/ (global_step + 1)})
        # wandb.log({"Total Temperature Violation in the episode": np.mean(total_temperature_violation)/ (global_step + 1)})
        # wandb.log({"Total Power Demand in the Episode": np.mean(total_power_demand)/ (global_step + 1)})
        
        for idx in range(num_tasks):
            model.replay_buffer.store(state[idx], action[idx].cpu().numpy(), reward[idx], next_state[idx], done[idx])

        state = next_state
        # Update model
        if step % 2048 == 0 and model.replay_buffer.size >= model.batch_size:
            # Sample a minibatch from replay buffer
            model.update()

        # if replay_buffer.size >= config.batch_size:
        #     for _ in range(config.updates_per_step):
        #         batch = replay_buffer.sample_batch()
        #         with torch.no_grad():
        #             next_state_action_logits = actor(batch['next_obs'])
        #             next_state_probs = F.softmax(next_state_action_logits, dim=-1)
        #             next_state_log_probs = F.log_softmax(next_state_action_logits, dim=-1)
        #             next_state_m = Categorical(probs=next_state_probs)
        #             next_state_actions = next_state_m.sample()
        #             next_q1_target, next_q2_target = critic_target(batch['next_obs'])
        #             next_q_target = torch.min(next_q1_target, next_q2_target)
        #             next_q = (next_state_probs * (next_q_target - log_alpha.exp() * next_state_log_probs)).sum(dim=-1)

        #             td_target = batch['rews'] + (1 - batch['done']) * gamma * next_q

        #         # Critic loss
        #         q1, q2 = critic(batch['obs'])
        #         q1_pred = q1.gather(1, batch['acts'].unsqueeze(-1)).squeeze(-1)
        #         q2_pred = q2.gather(1, batch['acts'].unsqueeze(-1)).squeeze(-1)
        #         critic_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        #         critic_optimizer.zero_grad()
        #         critic_loss.backward()
        #         critic_optimizer.step()

        #         # Actor loss
        #         state_action_logits = actor(batch['obs'])
        #         state_probs = F.softmax(state_action_logits, dim=-1)
        #         state_log_probs = F.log_softmax(state_action_logits, dim=-1)
        #         with torch.no_grad():
        #             q1_pi, q2_pi = critic(batch['obs'])
        #             min_q_pi = torch.min(q1_pi, q2_pi)
        #         actor_loss = (state_probs * (log_alpha.exp() * state_log_probs - min_q_pi)).sum(dim=1).mean()

        #         actor_optimizer.zero_grad()
        #         actor_loss.backward()
        #         actor_optimizer.step()

        #         # Entropy temperature adjustment
        #         alpha_loss = -(log_alpha * (state_log_probs.detach() + target_entropy).mean())
        #         alpha_optimizer.zero_grad()
        #         alpha_loss.backward()
        #         alpha_optimizer.step()

        #         # Soft update of target network
        #         for param, target_param in zip(critic.parameters(), critic_target.parameters()):
        #             target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)    

    # model_actor_loss.backward()
    # model_critic_loss.backward()    

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = 70
    model.save_model(f"sac_agent_checkpoint_{energy_weight}_{climate_type}.pth")
    # save_model(obs_dim, action_dim, config.learning_rate, model.replay_buffer, config.buffer_size, config.batch_size, gamma, tau, model.target_entropy, config.updates_per_step,\
    #                                    model.actor, model.critic, model.critic_target, model.actor_optimizer, model.critic_optimizer, model.log_alpha, model.alpha_optimizer, "sac_discrete_checkpoint.pth")
    for step in range(0, num_steps - 1):
        global_step += 1
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = model.choose_action(state_tensor)



        next_state, reward, done, truncated, info = envs.step(action.cpu().numpy())


        if 'total_power_demand' in info and 'total_temperature_violation' in info:
            total_power_demand = info["total_power_demand"]
            total_temperature_violation = info["total_temperature_violation"]
            temp_violation_list += total_temperature_violation
            power_demand_list += total_power_demand
            # Append episodic return to the list
            rewards_list += reward

            # for idx in range(num_tasks):
            #     replay_buffer.store(state[idx], action[idx].cpu().numpy(), reward[idx], next_state[idx], done[idx])

        else:
            print("Episode ended")
            print(info)

        state = next_state
    
    # Log the episodic return
    wandb.log({"Training Reward": np.mean(rewards_list) / (global_step + 1)})
    wandb.log({"Training Power Consumption":np.mean(power_demand_list)/ (global_step + 1)})
    wandb.log({"Training Temperature Violation":np.mean(temp_violation_list)/ (global_step + 1)})
    # wandb.log({"Total Temperature Violation in the episode": np.mean(total_temperature_violation)/ (global_step + 1)})
    # wandb.log({"Total Power Demand in the Episode": np.mean(total_power_demand)/ (global_step + 1)})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of the experiment")
    parser.add_argument("--gym-id", type=str, default="Eplus-A403-mixed-discrete-stochastic-v1",
                        help="the name of the gym environment")
    parser.add_argument("--algorithm-name", type=str, default="SAC",
                        help="the name of the algorithm for the agent")
    parser.add_argument("--climate-type", type=str, default="mixed",
                        help="the name of the environment climate")
    parser.add_argument("--learning-rate", type=float, default=0.0003,
                        help="the learning rate for the optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed of the experiment")
    parser.add_argument("--energy-weight", type=float, default=0.5,
                        help="energy weight for the reward function")
    parser.add_argument("--env-type", type=str, default='-stochastic-',
                        help="creates stochastic environment")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="the number of episodes")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will not be enabled by default")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel environments")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="the maximum size of the replay buffer")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="the batch size for sampling from the replay buffer")
    parser.add_argument("--updates-per-step", type=int, default=4,
                        help="number of updates per environment step")
    parser.add_argument("--lambda-temperature", type=float, default=1e-2,
                        help="lambda temperature for the reward function")
    parser.add_argument("--eval-mode", type=bool, default=False,
                        help="Activate fine-tuning and evaluation")
    parser.add_argument("--sweep-count", type=int, default=1, help="Count of the sweep")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)

    stochastic = args.env_type
    environment = f"Eplus-A403-mixed-discrete{stochastic}v1"
    # Name of the experiment
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = 'SAC' + '_' + str(args.energy_weight) + '_' + stochastic + '_' + args.climate_type
    
    name = experiment_name
    sweep_config = {
        'method': 'random', 
        'name': name
    }
    metric = {
        'name': 'Training Power Consumption',
        'goal': 'minimize'
    }
    parameters_dict = ({
        'learning_rate': {'value': args.learning_rate},
        'expname': {'value': args.expname},
        'gym_id': {'value': environment},
        'algorithm_name': {'value': args.algorithm_name},
        'seed': {'value': args.seed},
        'cuda': {'value': args.cuda},
        'num_envs': {'value': args.num_envs},
        'num_episodes': {'value': args.num_episodes},
        'buffer_size': {'value': args.buffer_size},
        'batch_size': {'value': args.batch_size},
        'updates_per_step': {'value': args.updates_per_step},
        'energy_weight': {'value': args.energy_weight},
        'climate_type': {'value': args.climate_type},
        'env_type': {'value': args.env_type},
        'lambda_temperature': {'value': 0.01},
        'eval_mode': {'value': args.eval_mode},
    })
    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metric

    sweep_id = wandb.sweep(sweep_config, project="SAC-TRAIN", entity="ulasfiliz")
    wandb.agent(sweep_id, train, count=args.sweep_count)