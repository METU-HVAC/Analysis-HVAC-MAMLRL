import sinergym
import gym
import torch
import wandb
import argparse
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import learn2learn as l2l
import time
import gymnasium as gym
from environment import *
from agents.models.SACAgent import SACDiscrete

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
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
        num_actions = 70
        num_states = np.array(envs.single_observation_space.shape).prod()
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        max_memory_length = 100000
        update_after_actions = 2048
        learning_rate = args.learning_rate_inner
        updates_per_step=4
        
        model = SACDiscrete(num_states, num_actions, updates_per_step, device, learning_rate, max_memory_length, batch_size).to(device)
        maml = l2l.algorithms.MAML(model, lr=config.learning_rate_inner, first_order=False, allow_unused=True)  
        
        opt = optim.Adam(maml.parameters(), config.learning_rate_outer)

        global_step = 0
        global_step_eval = 0
        
        observation,_ = envs.reset(seed=42)
        next_observations = torch.tensor(observation).to(device)
        num_updates = config.num_episodes
        meta_batch_size = 1
        adaptation_steps = 1
        start_time = time.time()

        meta_reward = [0.0] * num_tasks
        power_demand_train_episode = [0.0] * num_tasks    
        temp_violation_train_episode = [0.0] * num_tasks
        meta_actor_loss_global = 0.0
        meta_critic_loss_global = 0.0
        

        for update in range(0,num_updates):
            learner = maml.clone()
                
            power_demand_train_episode = [0.0] * num_tasks    
            temp_violation_train_episode = [0.0] * num_tasks
            for step in range(0,max_ep_len - 1):

                with torch.no_grad():
                    action = learner.module.choose_action(next_observations)
                    
                next_obs, reward,terminated,truncated,info = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
            
                # meta_reward += reward
                
                # if 'total_power_demand' in info:
                #     total_power_demand = info["total_power_demand"]
                #     power_demand_train_episode += total_power_demand    
                
                # if 'total_temperature_violation' in info: 
                #     total_temperature_violation = info["total_temperature_violation"]
                #     temp_violation_train_episode += total_temperature_violation
                
                # avg_reward = np.mean(meta_reward) / (global_step + 1) 
                # avg_power = np.mean(power_demand_train_episode) / (global_step + 1) 
                # avg_temperature = np.mean(temp_violation_train_episode) / (global_step + 1)  
                # run.log({"Reward": avg_reward})
                # run.log({"Average Power Consumption": avg_power,"Average Temperature Violation":avg_temperature})
                

                for idx in range(num_tasks):
                    # Store experiences in replay buffer
                    # Append episodic return to the list
                    learner.module.replay_buffer.store(next_observations[idx].cpu().numpy(), action[idx], reward[idx], next_obs[idx], done[idx])

                

                next_observations = torch.tensor(next_obs).to(device)

                # Update model
                if step % update_after_actions == 0 and learner.module.replay_buffer.size >= batch_size:
                    # Sample a minibatch from replay buffer
                    model_actor_loss, model_critic_loss = learner.module.update()
                    learner.adapt(0.5 * model_actor_loss + 0.5 * model_critic_loss)
                    meta_actor_loss_global -= model_actor_loss
                    meta_critic_loss_global -= model_critic_loss

                #Log the average episodic return every update
                if 'timestep' in info and len(info['timestep']) > 0 and step == max_ep_len:
                        wandb.log({"Final Power per Episode":np.mean(info["total_power_demand"]),"Final Temperature Violation per Episode":np.mean(info["total_temperature_violation"])}, step=global_step)     
                        
                        print("all environments are done")
                        next_observations,_ = envs.reset(seed=42)
                        next_observations = torch.tensor(next_observations).to(device)
                        print("Ended the whole episode.")
                        # latest_model_dir = f"{global_step}.pt"
                        # learner.module.save(latest_model_dir)
                            

            model_actor_loss.backward(retain_graph=True)
            model_critic_loss.backward(retain_graph=True)    
            #meta_actor_loss_global /= global_step_eval
            #meta_critic_loss_global /= global_step_eval
            opt.zero_grad()
            #meta_actor_loss_global.backward(retain_graph=True)
            #meta_critic_loss_global.backward(retain_graph=True)
            opt.step()

            for step in range(0,max_ep_len - 1):
                global_step += 1

                with torch.no_grad():
                    action = learner.module.choose_action(next_observations)
                    
                next_obs, reward,terminated,truncated,info = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
            
                meta_reward += reward
                
                if 'total_power_demand' in info:
                    total_power_demand = info["total_power_demand"]
                    power_demand_train_episode += total_power_demand    
                
                if 'total_temperature_violation' in info: 
                    total_temperature_violation = info["total_temperature_violation"]
                    temp_violation_train_episode += total_temperature_violation
                
                next_observations = torch.tensor(next_obs).to(device)

                #Log the average episodic return every update
                if 'timestep' in info and len(info['timestep']) > 0 and step == max_ep_len:
                        wandb.log({"Final Power per Episode":np.mean(info["total_power_demand"]),"Final Temperature Violation per Episode":np.mean(info["total_temperature_violation"])}, step=global_step)     
                        
                        print("all environments are done")
                        next_observations,_ = envs.reset(seed=42)
                        next_observations = torch.tensor(next_observations).to(device)
                        print("Ended the whole episode.")
                        # latest_model_dir = f"{global_step}.pt"
                        # learner.module.save(latest_model_dir)

            avg_reward = np.mean(meta_reward) / (global_step + 1) 
            avg_power = np.mean(power_demand_train_episode) / (global_step + 1) 
            avg_temperature = np.mean(temp_violation_train_episode) / (global_step + 1)  
            wandb.log({"Training Reward": avg_reward})
            wandb.log({"Training Power Consumption": avg_power,"Training Temperature Violation":avg_temperature})
                    
            wandb.log({"Meta Actor Loss": model_actor_loss / (global_step / 2048 + 1),
                "Meta Critic Loss": model_critic_loss / (global_step / 2048 + 1)})
        
        wandb.log({"Execution Time": abs(start_time - time.time())})
        model.save_model(f"MAML_sac_agent_checkpoint_{config.energy_weight}_{config.climate_type}.pth")
        torch.save({"meta_optimizer_state_dict": opt.state_dict()}, f"maml_sac_optimizer_{config.energy_weight}_{config.climate_type}.pth")
        envs.close()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-weight", type=float, default=0.01,
                        help="energy weight for the reward function")
    parser.add_argument("--env-type", type=str, default='-stochastic-',
                        help="creates stochastic environment")
    parser.add_argument("--climate-type", type=str, default='mixed',
                        help="creates climate of the environment")
    
    parser.add_argument("--learning-rate-inner", type=float, default=0.0003)
    parser.add_argument("--learning-rate-outer", type=float, default=0.0003)
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="energy weight for the reward function")
    parser.add_argument("--sweep-count", type=int, default=1, help="Count of the sweep")
    
    args = parser.parse_args()    
    environment = "Eplus-A403-mixed-discrete-stochastic-v1"
    # Name of the experiment
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = str(args.energy_weight) + '_' + 'stochastic' + '_' + 'MAML-SAC-' + environment + '-episodes-' + str(args.num_episodes) + '_' + str(args.climate_type) + '_' + args.climate_type
    experiment_name += '_' + experiment_date

    # Create wandb.config object in order to log all experiment params
    experiment_params = {
        'sinergym-version': sinergym.__version__,
        'python-version': sys.version
    }
    experiment_params.update({'environment': environment,
                            'episodes': args.num_episodes,
                            'algorithm': 'MAML-SAC'})

    # Get wandb init params (you have to specify your own project and entity)
    wandb_params = {"project": 'SAC-TRAIN',
                    "entity": 'ulasfiliz'}
    
    stochastic = args.env_type
    environment = f"Eplus-A403-mixed-discrete{stochastic}v1"
    # Name of the experiment
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = 'MAML-SAC' + str(args.energy_weight) + '_' + stochastic + '_' + str(args.climate_type)
    
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
        'learning_rate_inner': {'distribution': 'uniform',
                                'min': args.learning_rate_inner / 3,
                                'max': args.learning_rate_inner * 3},
        'learning_rate_outer': {'distribution': 'uniform',
                                'min': args.learning_rate_outer / 3,
                                'max': args.learning_rate_outer * 3},
        'expname': {'value': name},
        'gym_id': {'value': environment},
        'algorithm_name': {'value': "MAML-SAC"},
        'num_episodes': {'value': args.num_episodes},
        'energy_weight': {'value': args.energy_weight},
        'climate_type':  {'value': args.climate_type},
        'lambda_temperature': {'value': 1}
    })
    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metric

    sweep_id = wandb.sweep(sweep_config, project="SAC-TRAIN", entity="ulasfiliz")
    wandb.agent(sweep_id, train, count=args.sweep_count)    

# eval_obs,_ = eval_env.reset(seed=42*2)


# print("Resetting the evaluation environment")
# next_eval_obs = torch.tensor(eval_obs).to(device)
# eval_done = torch.zeros(1).to(device)

# for eval_step in range(max_ep_len):
#     global_step_eval += 1
#     with torch.no_grad():
#         eval_action = learner.module.choose_action(next_eval_obs)
        
#     eval_obs, eval_reward,eval_terminated,eval_truncated,eval_info = eval_env.step(eval_action.cpu().numpy())
#     eval_done = np.logical_or(eval_terminated, eval_truncated)
#     reward_eval_episode += eval_reward
    
#     if 'total_power_demand' in eval_info:
#         total_power_demand = eval_info["total_power_demand"]
#         power_demand_eval_episode += total_power_demand    
    
#     if 'total_temperature_violation' in eval_info: 
#         total_temperature_violation = eval_info["total_temperature_violation"]
#         temp_violation_eval_episode += total_temperature_violation
        
#     for idx in range(num_tasks):
#         # Store experiences in replay buffer
#         # Append episodic return to the list
#         learner.module.replay_buffer.store(eval_obs[idx], eval_action[idx], eval_reward[idx], next_eval_obs[idx].cpu().numpy(), eval_terminated[idx])
    
#     if eval_step % update_after_actions == 0 and learner.module.replay_buffer.size >= batch_size:
#         meta_actor_loss, meta_critic_loss = learner.module.update()

#     next_eval_obs , eval_done = torch.tensor(eval_obs).to(device) , torch.tensor(eval_done,dtype=torch.float32).to(device)

#     # run.log({"Meta Episodic Reward":np.mean(eval_reward)},step=global_step_eval)
#     # run.log({"Meta Episodic Power Consumption": np.mean(eval_info["total_power_demand"]),"Meta Episodic Temperature Violation":np.mean(eval_info["total_temperature_violation"])}, step=global_step_eval)     

#     avg_reward = np.mean(reward_eval_episode) / (global_step_eval + 1) 
#     avg_power = np.mean(power_demand_eval_episode) / (global_step_eval + 1) 
#     avg_temperature = np.mean(temp_violation_eval_episode) / (global_step_eval + 1)  

        
#     run.log({"Meta Reward": avg_reward},step=global_step_eval)
#     run.log({"Meta Average Power Consumption": avg_power,"Meta Average Temperature Violation":avg_temperature}, step=global_step_eval)         
    