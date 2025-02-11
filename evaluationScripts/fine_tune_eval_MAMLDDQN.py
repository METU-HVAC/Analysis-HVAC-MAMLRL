import torch
import random
import numpy as np
import wandb
import glob

import gymnasium as gym
from environment import *
from agents.models.DDQN_discrete import DDQNAgent
import learn2learn as l2l
import torch.optim as optim

cuda=True
device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

num_tasks = 1
max_ep_len = 52560

def fine_tune_policy(model, optimizer, eval_env, run, energy_weight, env_type, model_env_type, steps=20, batch_size = 64, lr=1e-3, meta_loss_global=0.0):
    """
    Fine-tunes the meta-trained policy in the meta-test environment.
    :param policy: The meta-trained policy.
    :param env: The meta-test environment.
    :param steps: Number of gradient steps for fine-tuning.
    :param lr: Learning rate for fine-tuning.
    :return: Adapted policy.
    """
    model_loss_global = 0
    rewards_list = [0.0]
    power_demand_list = [0.0]
    temp_violation_list = [0.0]
    global_step = 0
    for _ in range(steps):
        state = eval_env.reset(seed=42)
        next_observations, _ = state
        next_observations = torch.tensor(next_observations).to(device)
        done = False
        learner = model.clone()
        for step in range(0, max_ep_len - 1):
            # Predict action Q-values for all environments
            state_tensor = next_observations.to(device)  # Ensure state is on the correct device
            with torch.no_grad():
                action_probs = learner.module.policy(state_tensor)  # Forward pass through the model
                
            # Take the best action for each environment
            action = torch.argmax(action_probs, dim=1)
            
            next_obs, reward,terminated,truncated,info = eval_env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            # Record the experience in the replay buffer
            model.replay_buffer.store(next_observations.cpu().numpy(), action, reward, next_obs, done)

            
            # model.replay_buffer.store(next_observations.cpu().numpy(), action, reward, next_obs, done)
                    
            global_step += 1
            next_observations = torch.tensor(next_obs).to(device)
            if 'total_power_demand' in info and 'total_temperature_violation' in info:
                total_power_demand = info["total_power_demand"]
                total_temperature_violation = info["total_temperature_violation"]
                temp_violation_list += total_temperature_violation
                power_demand_list += total_power_demand
                # Append episodic return to the list
                rewards_list += reward
            
            if step % 2048 == 0 and learner.module.replay_buffer.size >= batch_size:
                model_loss = learner.module.update()
                learner.adapt(model_loss)
                model_loss_global -= model_loss

            # Log the episodic return
            # wandb.log({"Fine-Tuning Reward": np.mean(rewards_list) / (global_step + 1)}, step=global_step)
            # wandb.log({"Fine-Tuning Loss": np.mean(model_loss_global) / (global_step + 1)}, step=global_step)
            # wandb.log({"Fine-Tuning Power Consumption":np.mean(power_demand_list)/ (global_step + 1)}, step=global_step)
            # wandb.log({"Fine-Tuning Temperature Violation":np.mean(temp_violation_list)/ (global_step + 1)}, step=global_step)
    
            #Log the average episodic return every update
            if 'timestep' in info and len(info['timestep']) > 0 and step == max_ep_len - 1:
                    # run.log({"Final Power per Episode":np.mean(info["total_power_demand"]),"Final Temperature Violation per Episode":np.mean(info["total_temperature_violation"])})     
                    
                    # episodic_return = [0.0] * num_tasks
                    done = True
            
                    print("all environments are done")
        
        
        #meta_loss_global /= max_ep_len
        optimizer.zero_grad()
        model_loss.backward(retain_graph=True)  
        optimizer.step()

    # Save the agent's state
    model.save_ddqn_agent(f"maml_ddqn_fine_tuned_agent_weight_{energy_weight}_train_env_{model_env_type}_eval_env_{env_type}_episodes_{steps}.pth")
    torch.save({"meta_optimizer_state_dict": optimizer.state_dict()}, f"maml_ddqn_fine_tuned_optimizer_weight_{energy_weight}_train_env_{model_env_type}_eval_env_{env_type}.pth")
        

def evaluate_policy(model, eval_env, run, num_episodes=1):
    """
    Evaluates the fine-tuned policy in the meta-test environment.
    :param policy: The fine-tuned policy.
    :param env: The meta-test environment.
    :param num_episodes: Number of episodes for evaluation.
    :return: Average reward across episodes.
    """
    
    avg_reward = []
    avg_power = []
    avg_temperature = []
    episodic_return = 0.0
    power_demand_eval_episode = 0.0
    temp_violation_eval_episode = 0.0
    max_ep_len = 52560
    next_observations, info_state = eval_env.reset(seed=42)
    next_observations = torch.tensor(next_observations).to(device)
    eval_step = 0
    current_month = 0
    month_step = 0
    for _ in range(num_episodes):
        for step in range(0,max_ep_len):
            eval_step += 1 
            month_step += 1
            # Predict action Q-values for all environments
            state_tensor = next_observations.to(device)  # Ensure state is on the correct device
            with torch.no_grad():
                action_probs = model.policy(state_tensor)  # Forward pass through the model
            # Take the best action for each environment
            action = torch.argmax(action_probs, dim=1)
            
            # Decay probability of taking random action
            next_obs, reward,terminated,truncated,info = eval_env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            # Record the experience in the replay buffer
            model.replay_buffer.store(next_observations.cpu().numpy(), action, reward, next_obs, done)

            
            # model.replay_buffer.store(next_observations.cpu().numpy(), action, reward, next_obs, done)
            if 'total_power_demand' and 'total_temperature_violation' in info:
                total_power_demand = info["total_power_demand"]
                total_temperature_violation = info["total_temperature_violation"]
                    
                power_demand_eval_episode += total_power_demand    
                temp_violation_eval_episode += total_temperature_violation

            #run.log({"Power Consumption Per Episode": np.mean([x/(step + 1) for x in power_demand_train_episode]),"Temperature Violation Per Episode":np.mean([x/(step + 1) for x in temp_violation_train_episode])}, step=global_step)  
            
            next_observations , next_dones = torch.tensor(next_obs).to(device) , torch.tensor(done,dtype=torch.float32).to(device)
            episodic_return += reward  # Convert the numpy value to a Python float
            
            #Log the average episodic return every update
            if 'timestep' in info and len(info['timestep']) > 0 and step == max_ep_len - 1:
                    # run.log({"Final Power per Episode":np.mean(info["total_power_demand"]),"Final Temperature Violation per Episode":np.mean(info["total_temperature_violation"])})     
                    
                    # episodic_return = [0.0] * num_tasks
                    next_done = torch.zeros(num_tasks).to(device)

            if current_month != info["month"]:
                if current_month == 12 and info["month"] == 1:
                    current_month = [13]
                else:
                    current_month = info["month"]
                # Log the episodic return
                if current_month[0] != 1:
                    run.log({"Evaluation Reward": np.mean(episodic_return) / (month_step + 1)}, step=current_month[0])
                    run.log({"Evaluation Power Consumption":np.mean(power_demand_eval_episode)/ (month_step + 1)}, step=current_month[0])
                    run.log({"Evaluation Temperature Violation":np.mean(temp_violation_eval_episode)/ (month_step + 1)}, step=current_month[0])
                month_step = 0
                episodic_return = 0.0
                power_demand_eval_episode = 0.0
                temp_violation_eval_episode = 0.0
        # avg_reward = np.mean(episodic_return) / (eval_step + 1) 
        # avg_power = np.mean(power_demand_eval_episode) / (eval_step + 1) 
        # avg_temperature = np.mean(temp_violation_eval_episode) / (eval_step + 1)  
        
        # run.log({"Evaluation Reward": avg_reward})
        # run.log({"Evaluation Power Consumption": avg_power,
        #             "Evaluation Temperature Violation":avg_temperature})

def main(args):

    num_actions = 70
    episodes = 50
    adapt_lr=0.01
    meta_lr=0.001
    climate = args.env_type
    
    environment = f"Eplus-A403-{climate}-discrete-stochastic-v1"
    # Name of the experiment
    experiment_name = f'MAML-DDQN-' + str(args.energy_weight) + '_' + climate + '_' + str(args.fine_tune) + '_' + args.model_env_type

    # Create wandb.config object in order to log all experiment params
    experiment_params = {
        'sinergym-version': sinergym.__version__,
        'python-version': sys.version
    }
    experiment_params.update({'environment': environment,
                            'episodes': episodes,
                            'algorithm': "DDQN"})

    # Get wandb init params (you have to specify your own project and entity)
    wandb_params = {"project": "DDQN-Eval",
                    "entity": 'ulasfiliz'}
    # Init wandb entry
    run = wandb.init(
        name=experiment_name,
        config=experiment_params,
        **wandb_params
    )
    
    # Initialize the environment
    eval_env_fine_tune = gym.vector.SyncVectorEnv([make_env(
        gym_id=environment,
            seed=42,energy_weight=args.energy_weight,lambda_temperature=1,obs_to_remove=[])])
    eval_env_eval = gym.vector.SyncVectorEnv([make_env(
        gym_id=environment,
            seed=42,energy_weight=args.energy_weight,lambda_temperature=1,obs_to_remove=[])])

    
    batch_size = 64

    model = DDQNAgent(num_states=np.array(eval_env_fine_tune.observation_space.shape).prod(), num_actions=num_actions, batch_size=batch_size, device=device).to(device)

    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    # Load meta-trained parameters
    optimizer_path = f'/home/ulas_filiz/git/sinergym/maml_ddqn_optimizer_{args.energy_weight}_{args.model_env_type}.pth' 
    checkpoint = torch.load(optimizer_path, map_location=device)
    
    # Load model state
    maml.module.load_ddqn_agent(f"maml_ddqn_agent_{args.energy_weight}_{args.model_env_type}.pth", device=device)
    opt.load_state_dict(checkpoint["meta_optimizer_state_dict"])
    
    if not args.use_fine_tuned_model and not (f"maml_ddqn_fine_tuned_agent_weight_{args.energy_weight}_train_env_{args.model_env_type}_eval_env_{args.env_type}.pth" in glob.glob("~/git/sinergym")
                                           and f"maml_ddqn_fine_tuned_optimizer_weight_{args.energy_weight}_train_env_{args.model_env_type}_eval_env_{args.env_type}.pth" in glob.glob("~/git/sinergym")):  
        if args.fine_tune :
            print("Tuning: ", args.fine_tune)
            model.train()
            fine_tune_policy(maml, opt, eval_env_fine_tune,run, energy_weight=args.energy_weight, env_type=args.env_type, model_env_type=args.model_env_type)
    else:
        # Load meta-trained parameters
        model.load_ddqn_agent(f"maml_ddqn_fine_tuned_agent_weight_{args.energy_weight}_train_env_{args.model_env_type}_eval_env_{args.env_type}.pth", device="cuda")
        opt.load_state_dict(torch.load(f"maml_ddqn_fine_tuned_optimizer_weight_{args.energy_weight}_train_env_{args.model_env_type}_eval_env_{args.env_type}.pth", map_location=device)["meta_optimizer_state_dict"])
    
    model.eval()
    evaluate_policy(model, eval_env_eval, run)
    # Evaluate the fine-tuned policy
    
    eval_env_fine_tune.close()
    eval_env_eval.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-weight", type=float, default=0.5,
                        help="energy weight for the reward function")
    parser.add_argument("--env-type", type=str, default='hot',
                        help="creates environment with a certain climate")
    parser.add_argument("--model-env-type", type=str, default='mixed',
                        help="calls the modelo of a certain climate")
    parser.add_argument("--model-path", type=str, default='/home/ulas_filiz/git/sinergym/maml_ddqn_optimizer.pth',
                        help="Name of the evaluated model's checkpoint.")
    parser.add_argument("--fine-tune", type=bool, default=False,
                        help="Activate fine-tuning and evaluation")
    parser.add_argument("--use-fine-tuned-model", type=bool, default=False,
                        help="utilize a fine-tuned model if it exists.")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="energy weight for the reward function")
    args = parser.parse_args()    
    main(args)