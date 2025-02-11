#!/usr/bin/env python3

import traceback
from torch.autograd import grad
import torch.nn as nn
import numpy as np
import torch
import random
import torch.optim as optim
from torch.distributions.categorical import Categorical

import gymnasium as gym
from environment import *
from agents.models.DDQN_discrete import DDQNAgent
import time

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        stochastic = args.env_type
        cuda=True
        device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        # Observation dimensions
        all_obs_list = ['month', 'day_of_month', 'hour', 'outdoor_temperature', 'outdoor_humidity', 
        'wind_speed', 'wind_direction', 'diffuse_solar_radiation', 'direct_solar_radiation', 
        'htg_setpoint', 'clg_setpoint', 'air_temperature', 'air_humidity', 'people_occupant', 
        'co2_emission', 'HVAC_electricity_demand_rate', 'total_electricity_HVAC']
        obs_remove = ['co2_emission']
        #obs_remove = ['wind_speed', 'wind_direction', 'diffuse_solar_radiation', 'direct_solar_radiation']
        #obs_remove = ['htg_setpoint', 'clg_setpoint','wind_speed', 'wind_direction', 'diffuse_solar_radiation', 'direct_solar_radiation','outdoor_humidity','air_humidity']
        #obs_remove = ['total_electricity_HVAC','month', 'day_of_month', 'hour']
        #obs_remove = []
        OBS_VARIABLES = [obs for obs in all_obs_list if obs not in obs_remove]
        #Observation dimensions
        OBSERVATION_DIM = len(OBS_VARIABLES)
        max_ep_len = 52560
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
            energy_weight=args.energy_weight,lambda_temperature=1,obs_to_remove=[]) 
            )
        envs = gym.vector.AsyncVectorEnv(envs_list)

        batch_size = 64
        model = DDQNAgent(num_states=np.array(envs.single_observation_space.shape).prod(), num_actions=70, batch_size=batch_size, device=device,lr=config.learning_rate).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        latest_model_dir = ""
        global_step = 0
        meta_train_loss = torch.tensor(0.0, device=device, requires_grad=True) 
        observation,_ = envs.reset(seed=42)
        next_observations = torch.tensor(observation).to(device)
        next_done = torch.zeros(num_tasks).to(device)
        num_updates = config.num_episodes
        adaptation_steps = 1
        num_actions = 70
        episodic_return = [0.0] * num_tasks
        power_demand_train_episode = [0.0] * num_tasks    
        temp_violation_train_episode = [0.0] * num_tasks

        meta_reward = [0.0] * num_tasks
        power_demand_eval_episode = [0.0] * num_tasks    
        temp_violation_eval_episode = [0.0] * num_tasks

        anneal_lr = True
        global_step_eval = 0
        epsilon_greedy_steps = 1000000.0
        epsilon = 0.1  # Epsilon greedy parameter
        epsilon_min = 0.1  # Minimum epsilon greedy parameter
        epsilon_max = 1.0  # Maximum epsilon greedy parameter
        epsilon_interval = (
            epsilon_max - epsilon_min
        )  # Rate at which to reduce chance of random action being taken
        
        model_loss = 0.0
        model_loss_global = 0.0

        power_demand_train_episode = [0.0] * num_tasks    
        temp_violation_train_episode = [0.0] * num_tasks
        start_time = time.time() 
        for update in range(0,num_updates):
            for step in range(0,max_ep_len - 1):
                if epsilon > random.random():
                        # Take random actions for each environment
                        action = torch.randint(0, num_actions, (num_tasks,), device=device)
                else:
                    # Predict action Q-values for all environments
                    state_tensor = next_observations.to(device)  # Ensure state is on the correct device
                    with torch.no_grad():
                        action_probs = model.policy(state_tensor)  # Forward pass through the model
                    # Take the best action for each environment
                    action = torch.argmax(action_probs, dim=1)
                
                # Decay probability of taking random action
                epsilon -= epsilon_interval / epsilon_greedy_steps
                epsilon = max(epsilon, epsilon_min)
                next_obs, reward,terminated,truncated,info = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                
                # Record the experience in the replay buffer
                for idx in range(num_tasks):
                    # Store experiences in replay buffer
                    # Append episodic return to the list
                    model.replay_buffer.store(next_observations[idx].cpu().numpy(), action[idx], reward[idx], next_obs[idx], done[idx])

                
                # model.replay_buffer.store(next_observations.cpu().numpy(), action, reward, next_obs, done)
                if 'total_power_demand' in info and 'total_temperature_violation' in info:
                    total_power_demand = info["total_power_demand"]
                    total_temperature_violation = info["total_temperature_violation"]
                        
                # power_demand_train_episode += total_power_demand    
                # temp_violation_train_episode += total_temperature_violation
                #run.log({"Power Consumption Per Episode": np.mean([x/(step + 1) for x in power_demand_train_episode]),"Temperature Violation Per Episode":np.mean([x/(step + 1) for x in temp_violation_train_episode])}, step=global_step)  
                
                next_observations , next_dones = torch.tensor(next_obs).to(device) , torch.tensor(done,dtype=torch.float32).to(device)
                #episodic_return += reward  # Convert the numpy value to a Python float
                
                # avg_reward = np.mean(episodic_return) / (global_step + 1) 
                # avg_power = np.mean(power_demand_train_episode) / (global_step + 1) 
                # avg_temperature = np.mean(temp_violation_train_episode) / (global_step + 1)  
                
                # run.log({"Meta Reward": avg_reward},step=global_step)
                # run.log({"Meta Average Power Consumption": avg_power,
                #             "Meta Average Temperature Violation":avg_temperature}, step=global_step)

                if step % 2048 == 0 and model.replay_buffer.size >= batch_size:
                    model_loss = model.update()
                    #model_loss_global -= model_loss
                    # run.log({"Meta Loss": model_loss_global / (global_step_eval / 4 + 1)}, step=global_step)
                    
                # run.log({"episodic_return":np.mean([x/(step + 1) for x in episodic_return])},step=global_step)
                
                #Log the average episodic return every update
                if 'timestep' in info and len(info['timestep']) > 0 and step == max_ep_len - 1:
                        # run.log({"Final Power per Episode":np.mean(info["total_power_demand"]),"Final Temperature Violation per Episode":np.mean(info["total_temperature_violation"])})     
                        
                        # episodic_return = [0.0] * num_tasks
                        next_observations,_ = envs.reset(seed=42)
                        next_observations = torch.tensor(next_observations).to(device)
                        next_done = torch.zeros(num_tasks).to(device)
                
                        print("all environments are done")
                
            # Save the agent's state
            model.save_ddqn_agent(f"ddqn_agent_state_{config.energy_weight}_{config.climate_type}.pth")

            # Agent Evaluation
            for step in range(0,max_ep_len - 1):
                global_step += 1
                # Predict action Q-values for all environments
                state_tensor = next_observations.to(device)  # Ensure state is on the correct device
                with torch.no_grad():
                    action_probs = model.policy(state_tensor)  # Forward pass through the model
                # Take the best action for each environment
                action = torch.argmax(action_probs, dim=1)
            
                next_obs, reward,terminated,truncated,info = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                
                # # Record the experience in the replay buffer
                # for idx in range(num_tasks):
                #     # Store experiences in replay buffer
                #     # Append episodic return to the list
                #     model.replay_buffer.store(next_observations[idx].cpu().numpy(), action[idx], reward[idx], next_obs[idx], done[idx])

                
                # model.replay_buffer.store(next_observations.cpu().numpy(), action, reward, next_obs, done)
                if 'total_power_demand' in info and 'total_temperature_violation' in info:
                        
                    total_power_demand = info["total_power_demand"]
                    total_temperature_violation = info["total_temperature_violation"]
                            
                    power_demand_train_episode += total_power_demand    
                    temp_violation_train_episode += total_temperature_violation
                    #run.log({"Power Consumption Per Episode": np.mean([x/(step + 1) for x in power_demand_train_episode]),"Temperature Violation Per Episode":np.mean([x/(step + 1) for x in temp_violation_train_episode])}, step=global_step)  
                    
                    episodic_return += reward  # Convert the numpy value to a Python float
                next_observations , next_dones = torch.tensor(next_obs).to(device) , torch.tensor(done,dtype=torch.float32).to(device)
                    
                # run.log({"episodic_return":np.mean([x/(step + 1) for x in episodic_return])},step=global_step)
                
                #Log the average episodic return every update
                if 'timestep' in info and len(info['timestep']) > 0 and step == max_ep_len - 1:
                        # run.log({"Final Power per Episode":np.mean(info["total_power_demand"]),"Final Temperature Violation per Episode":np.mean(info["total_temperature_violation"])})     
                        
                        # episodic_return = [0.0] * num_tasks
                        next_observations,_ = envs.reset(seed=42)
                        next_observations = torch.tensor(next_observations).to(device)
                        next_done = torch.zeros(num_tasks).to(device)
                
                        print("all environments are done")
            
            avg_reward = np.mean(episodic_return) / (global_step + 1) 
            avg_power = np.mean(power_demand_train_episode) / (global_step + 1) 
            avg_temperature = np.mean(temp_violation_train_episode) / (global_step + 1)  
            
            wandb.log({"Training Reward": avg_reward})
            wandb.log({"Training Power Consumption": avg_power,
                        "Training Temperature Violation":avg_temperature})
                
        wandb.log({"Execution Time": abs(start_time - time.time())})
        envs.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-weight", type=float, default=0.01,
                        help="energy weight for the reward function")
    parser.add_argument("--env-type", type=str, default='-stochastic-',
                        help="creates stochastic environment")
    parser.add_argument("--climate-type", type=str, default='mixed',
                        help="creates climate environment")
    
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="energy weight for the reward function")
    parser.add_argument("--sweep-count", type=int, default=1, help="Count of the sweep")
    
    args = parser.parse_args()    
    environment = "Eplus-A403-mixed-discrete-stochastic-v1"
    # Name of the experiment
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = str(args.energy_weight) + '_' + 'stochastic' + '_' + 'DDQN-' + environment + '-episodes-' + str(args.num_episodes) + '_' + args.climate_type
    experiment_name += '_' + experiment_date

    # Create wandb.config object in order to log all experiment params
    experiment_params = {
        'sinergym-version': sinergym.__version__,
        'python-version': sys.version
    }
    experiment_params.update({'environment': environment,
                            'episodes': args.num_episodes,
                            'algorithm': 'DDQN'})

    # Get wandb init params (you have to specify your own project and entity)
    wandb_params = {"project": 'DDQN-TRAIN',
                    "entity": 'ulasfiliz'}
    
    stochastic = args.env_type
    environment = f"Eplus-A403-mixed-discrete{stochastic}v1"
    # Name of the experiment
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = 'DDQN' + "_" + str(args.energy_weight) + '_' + stochastic
    
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
        'expname': {'value': name},
        'gym_id': {'value': environment},
        'algorithm_name': {'value': "DDQN"},
        'num_episodes': {'value': args.num_episodes},
        'energy_weight': {'value': args.energy_weight},
        'climate_type': {'value': args.climate_type},
        'learning_rate': {'value': args.learning_rate},
        # 'learning_rate': {
        #     'distribution': 'uniform',
        #     'min': args.learning_rate / 3,
        #     'max': args.learning_rate * 3
        # },
        'lambda_temperature': {'value': 1}
    })
    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metric

    sweep_id = wandb.sweep(sweep_config, project="DDQN-TRAIN", entity="ulasfiliz")
    wandb.agent(sweep_id, train, count=args.sweep_count)