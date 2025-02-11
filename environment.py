import gymnasium as gym
import numpy as np
import sys
from datetime import datetime
import sinergym
from sinergym.utils.wrappers import (NormalizeAction,NormalizeObservation)
import wandb
import torch
from sinergym.utils.rewards import LinearReward
from math import exp
from typing import Any, Dict, List, Tuple, Union
from sinergym.utils.constants import LOG_REWARD_LEVEL, YEAR
from sinergym.utils.wrappers import (NormalizeAction,NormalizeObservation,ReduceObservationWrapper)

WANDB_ENTITY = 'ulasfiliz'
WADNB_PROJECT = 'RL-sinergym'
comfort_term_arr = []
energy_term_arr = []
class MyCustomReward(LinearReward):
    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        
        super(LinearReward, self).__init__()
        # Name of the variables
        self.temp_names = temperature_variables
        self.energy_names = energy_variables

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Summer period
        self.summer_start = summer_start  # (month,day)
        self.summer_final = summer_final  # (month,day)

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Check variables to calculate reward are available
        try:
            assert all(temp_name in list(obs_dict.keys())
                       for temp_name in self.temp_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the temperature variables specified are not present in observation.')
            raise err
        try:
            assert all(energy_name in list(obs_dict.keys())
                       for energy_name in self.energy_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the energy variables specified are not present in observation.')
            raise err
        # Get the number of people in the room (occupancy)
        people_occupant = obs_dict['people_occupant']
        # Energy calculation
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values,people_occupant)

        # Comfort violation calculation
        total_temp_violation, temp_violations = self._get_temperature_violation(
            obs_dict)
        comfort_penalty = self._get_comfort_penalty(temp_violations)

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward(
            energy_penalty, comfort_penalty)

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'reward_weight': self.W_energy,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': comfort_penalty,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': total_temp_violation
        }
        
        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> Tuple[float,
                                                                 List[float]]:
        """Calculate the total energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            Tuple[float, List[float]]: Total energy consumed (sum of variables) and List with energy consumed in each energy variable.
        """

        energy_values = [
            v for k, v in obs_dict.items() if k in self.energy_names]

        # The total energy is the sum of energies
        total_energy = sum(energy_values)

        return total_energy, energy_values

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
            Tuple[float, List[float]]: Total temperature violation (ºC) and list with temperature violation in each zone.
        """

        month = obs_dict['month']
        day = obs_dict['day_of_month']
        year = YEAR
        current_dt = datetime(int(year), int(month), int(day))

        # Periods
        summer_start_date = datetime(
            int(year),
            self.summer_start[0],
            self.summer_start[1])
        summer_final_date = datetime(
            int(year),
            self.summer_final[0],
            self.summer_final[1])

        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        temp_values = [
            v for k, v in obs_dict.items() if k in self.temp_names]
        total_temp_violation = 0.0
        temp_violations = []
        person_count = obs_dict['people_occupant']
        for T in temp_values:
            if T < temp_range[0] or T > temp_range[1]:
                temp_violation = min(
                    abs(temp_range[0] - T), abs(T - temp_range[1]))
                if person_count==0 :
                    # if there is no person in the room, the comfort violation is not considered 
                    break
                temp_violations.append(temp_violation)
                total_temp_violation += temp_violation
        
        
        return total_temp_violation, temp_violations

    def _get_energy_penalty(self, energy_values: List[float],occupancy: int) -> float:
        """Calculate the negative absolute energy penalty based on energy values

        Args:
            energy_values (List[float]): Energy values

        Returns:
            float: Negative absolute energy penalty value
        """
        
        total_energy = sum(energy_values)
    
        if occupancy == 0:
            # Penalize energy consumption more heavily when the room is empty
            energy_penalty = -10*total_energy  # Double the penalty when no people are present
        else:
            # Normal energy penalty
            energy_penalty = -total_energy   
        return energy_penalty

    def _get_comfort_penalty(self, temp_violations: List[float]) -> float:
        """Calculate the negative absolute comfort penalty based on temperature violation values

        Args:
            temp_violations (List[float]): Temperature violation values

        Returns:
            float: Negative absolute comfort penalty value
        """
        comfort_penalty = -sum(temp_violations)
        return comfort_penalty

    def _get_reward(self, energy_penalty: float,
                    comfort_penalty: float) -> Tuple[float, float, float]:
        """It calculates reward value using the negative absolute comfort and energy penalty calculates previously.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float,float,float]: total reward calculated, reward term for energy, reward term for comfort.
        """
        
        energy_term = self.lambda_energy * self.W_energy * energy_penalty
        comfort_term = self.lambda_temp * \
            (1 - self.W_energy) * comfort_penalty
        # comfort_term_arr.append(comfort_term)
        # energy_term_arr.append(energy_term)
        # print("comfort reward:",np.mean(comfort_term_arr),", energy term: ",np.mean(energy_term_arr))
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term

def create_experiment_name(env_name, episodes,algorithm_name):

    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = algorithm_name+'-' + env_name + \
        '-episodes-' + str(episodes)
    experiment_name += '_' + experiment_date

    return experiment_name

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
        env = ReduceObservationWrapper(env=env, obs_reduction=obs_to_remove)
        env = NormalizeObservation(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def create_environment(env_name, episodes,algorithm_name,lambda_temperature=1.0,energy_weight=0.6):
    # Create the experiment name
    experiment_name = create_experiment_name(env_name, episodes,algorithm_name)

    reward_kwargs = {
        "temperature_variables" : ["air_temperature"],
        "energy_variables"      : ["HVAC_electricity_demand_rate"],
        "range_comfort_winter"  : [20.0, 23.5],
        "range_comfort_summer"  : [23.0, 26.0],
        "summer_start"          : [6, 1],
        "summer_final"          : [9, 30],
        "energy_weight"         : energy_weight,
        "lambda_energy"         : 1e-4,
        "lambda_temperature"    : 1
        }
    #Create the environment
    env = gym.make(env_name,env_name=experiment_name, reward=MyCustomReward, reward_kwargs=reward_kwargs)
    # # Create the environment
    # env = gym.make(env_name,env_name=experiment_name, reward=LinearReward, reward_kwargs={
    #     "temperature_variables" : ["air_temperature"],
    #     "energy_variables"      : ["HVAC_electricity_demand_rate"],
    #     "range_comfort_winter"  : [20.0, 23.5],
    #     "range_comfort_summer"  : [23.0, 26.0],
    #     "summer_start"          : [6, 1],
    #     "summer_final"          : [9, 30],
    #     "energy_weight"         : energy_weight,
    #     "lambda_energy"         : 1e-4,
    #     "lambda_temperature"    : 1.0
    #     })
    # Create the environment
    env = gym.make(env_name,env_name=experiment_name, reward=LinearReward, reward_kwargs=reward_kwargs)

    return env,experiment_name

def close_environment(env):
    # Close the environment
    env.close()

def initialize_wandb(env_name, episodes,algorithm_name,experiment_name):
    # Create wandb.config object in order to log all experiment params
    experiment_params = {
        'sinergym-version': sinergym.__version__,
        'python-version': sys.version
    }
    experiment_params.update({'environment':env_name,
                            'episodes':episodes,
                            'algorithm':algorithm_name})

    # Get wandb init params (you have to specify your own project and entity)
    wandb_params = {"project": 'RL-sinergym',
                    "entity": 'ulasfiliz'}
    # Init wandb entry
    run = wandb.init(
        name=experiment_name + '_' + wandb.util.generate_id(),
        config=experiment_params,
        ** wandb_params
    )
    return run

def close_wandb(wandb_run,env):
    artifact = wandb.Artifact(
        name="experiment1",
        type="training")
    artifact.add_dir(
            env.get_wrapper_attr('workspace_path'),
            name='training_output/')
    wandb_run.log_artifact(artifact)

    # wandb has finished
    wandb_run.finish()