import torch

import gymnasium as gym
from environment import *
import torch.optim as optim
from torch.distributions import Categorical
import wandb
import argparse
import torch.nn.functional as F
from agents.models.SACAgent import SACDiscrete
import learn2learn as l2l
import glob
cuda=True
device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

num_tasks = 1
eval_step = 0
max_ep_len = 52560
meta_actor_loss_global = 0.0
meta_critic_loss_global = 0.0

def evaluate_sac(eval_env, model, run):
    model.eval()
    global_step = 0
    rewards_list = [0.0]
    power_demand_list = [0.0]
    temp_violation_list = [0.0]
    current_month = 0 
    month_step = 0
    state, _ = eval_env.reset()
    for _ in range(1):
        for step in range(0, max_ep_len - 1):
            global_step += 1
            month_step += 1
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                logits = model.actor(state_tensor)
                action = torch.argmax(logits, dim=1)  # actions shape: [num_envs]
                # probs = F.softmax(logits, dim=-1)
                # m = Categorical(probs)
                # action = m.sample()
                
            next_state, reward, done, truncated, info = eval_env.step(action.cpu().numpy())

            if 'total_power_demand' in info and 'total_temperature_violation' in info:
                total_power_demand = info["total_power_demand"]
                total_temperature_violation = info["total_temperature_violation"]
                temp_violation_list += total_temperature_violation
                power_demand_list += total_power_demand
                # Append episodic return to the list
                rewards_list += reward

            if current_month != info["month"]:
                if current_month == 12 and info["month"] == 1:
                    current_month = [13]
                else:
                    current_month = info["month"]
                # Log the episodic return
                if current_month[0] != 1:
                    run.log({"Evaluation Reward": np.mean(rewards_list) / (month_step + 1)}, step=current_month[0])
                    run.log({"Evaluation Power Consumption":np.mean(power_demand_list)/ (month_step + 1)}, step=current_month[0])
                    run.log({"Evaluation Temperature Violation":np.mean(temp_violation_list)/ (month_step + 1)}, step=current_month[0])
                month_step = 0
                rewards_list = 0.0
                power_demand_list = 0.0
                temp_violation_list = 0.0
            state = next_state
        
        

def train_sac(envs, model, opt, device, energy_weight, env_type, model_env_type, steps=20):
    global_step = 0
    state, _ = envs.reset()
    model.train()
    rewards_list = [0.0]
    power_demand_list = [0.0]
    temp_violation_list = [0.0]

    for _ in range(0,steps):
        for step in range(0, max_ep_len - 1):
            global_step += 1
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                logits = model.actor(state_tensor)
                action = torch.argmax(logits, dim=1)  # actions shape: [num_envs]
                # probs = F.softmax(logits, dim=-1)
                # m = Categorical(probs)
                # action = m.sample()
                
                next_state, reward, done, truncated, info = envs.step(action.cpu().numpy())

            model.replay_buffer.store(state, action.cpu().numpy(), reward, next_state, done)
            if 'total_power_demand' in info and 'total_temperature_violation' in info:
                total_power_demand = info["total_power_demand"]
                total_temperature_violation = info["total_temperature_violation"]
                temp_violation_list += total_temperature_violation
                power_demand_list += total_power_demand
                # Append episodic return to the list
                rewards_list += reward
            
            # Log the episodic return
            # wandb.log({"Fine-Tuning Reward": np.mean(rewards_list) / (global_step + 1)}, step=global_step)
            # wandb.log({"Fine-Tuning Power Consumption":np.mean(power_demand_list)/ (global_step + 1)}, step=global_step)
            # wandb.log({"Fine-Tuning Temperature Violation":np.mean(temp_violation_list)/ (global_step + 1)}, step=global_step)
            state = next_state

        
            # Update model
            if step % 2048 == 0 and model.replay_buffer.size >= 64:
                # Sample a minibatch from replay buffer
                model_actor_loss, model_critic_loss = model.update()
                meta_actor_loss_global -= model_actor_loss
                meta_critic_loss_global -= model_critic_loss

        # meta_actor_loss_global /= global_step
        # meta_critic_loss_global /= global_step
        opt.zero_grad()
        model_actor_loss.backward(retain_graph=True)
        model_critic_loss.backward(retain_graph=True)
        opt.step()

        # Save the agent's state
        model.save_model(f"sac_fine_tuned_agent_weight_{energy_weight}_train_env_{model_env_type}_eval_env_{env_type}_episodes_{steps}.pth")

def main(args):

    num_actions = 70
    episodes = 5
    climate = args.env_type
    environment = f"Eplus-A403-mixed-discrete-v1"
    # Name of the experiment
    experiment_name = f'SAC-' + str(args.energy_weight) + '_' + climate + '_' + str(args.fine_tune) + '_' + args.model_env_type

    # Create wandb.config object in order to log all experiment params
    experiment_params = {
        'sinergym-version': sinergym.__version__,
        'python-version': sys.version
    }
    experiment_params.update({'environment': environment,
                            'episodes': episodes,
                            'algorithm': "SAC-Eval"})

    # Get wandb init params (you have to specify your own project and entity)
    wandb_params = {"project": "SAC-Eval",
                    "entity": 'ulasfiliz'}
    # Init wandb entry
    run = wandb.init(
        name=experiment_name,
        config=experiment_params,
        **wandb_params
    )
    
    # Initialize the environment
    eval_env = gym.vector.SyncVectorEnv([make_env(
        gym_id=environment,
            seed=42,energy_weight=args.energy_weight,lambda_temperature=1,obs_to_remove=[])])

    
    batch_size = 64
    adapt_lr=0.01
    meta_lr=0.001

    
    model = SACDiscrete(obs_dim=np.array(eval_env.observation_space.shape).prod(), action_dim=num_actions, updates_per_step=4, buffer_size=1000000, learning_rate=adapt_lr, batch_size=batch_size, device=device).to(device)

    # Load meta-trained parameters
    model.load_model(f"sac_agent_checkpoint_{args.energy_weight}_{args.model_env_type}.pth")
    
    optimizer_path = f'/home/ulas_filiz/git/sinergym/maml_sac_optimizer_{args.energy_weight}_{args.model_env_type}.pth'
    checkpoint = torch.load(optimizer_path, map_location=device)
    opt = optim.Adam(model.parameters(), meta_lr)

    meta_actor_loss_global, meta_critic_loss_global = 0.0, 0.0 
    if not args.use_fine_tuned_model and not f"sac_fine_tuned_agent_weight_{args.energy_weight}_train_env_{args.model_env_type}_eval_env_{args.env_type}.pth" in glob.glob("~/git/sinergym"):        
        if args.fine_tune:
            train_sac(eval_env, model, opt, device, meta_actor_loss_global, meta_critic_loss_global, energy_weight=args.energy_weight, env_type=args.env_type, model_env_type=args.model_env_type)
    else:
        # Load meta-trained parameters
        model.load_ddqn_agent(f"sac_fine_tuned_agent_weight_{args.energy_weight}_train_env_{args.model_env_type}_eval_env_{args.env_type}.pth", device="cuda")

    # Evaluate the fine-tuned policy
    model.eval()
    evaluate_sac(eval_env, model, run)

    eval_env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-weight", type=float, default=0.5,
                        help="energy weight for the reward function")
    parser.add_argument("--env-type", type=str, default='hot',
                        help="creates environment with a certain climate")
    parser.add_argument("--model-env-type", type=str, default='mixed',
                        help="calls the modelo of a certain climate")
    parser.add_argument("--model-path", type=str, default="maml_sac_optimizer.pth",
                        help="Name of the evaluated model's checkpoint.")
    parser.add_argument("--fine-tune", type=bool, default=False,
                        help="Activate fine-tuning and evaluation")
    parser.add_argument("--use-fine-tuned-model", type=bool, default=False,
                        help="utilize a fine-tuned model if it exists.")
    args = parser.parse_args()    
    main(args)