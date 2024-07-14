import mlagents
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel # infos : https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Python-LLAPI.md
from mlagents_envs.environment import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

import numpy as np
import torch

def load_env(path,ts=20,decrease_reward_on_stay = 0, coin_visible = 1, obstacle_proba = 0.3,coin_proba = 1.0, move_speed = 1, turn_speed = 150, momentum = 0): #dont forget to close it before opening a new one
    try:
        global env
        env.close()
    except:
        env = None
        
    channel_engine = EngineConfigurationChannel()
    channel_env = EnvironmentParametersChannel()

    env = UnityEnvironment(path,side_channels=[channel_engine,channel_env])
    channel_engine.set_configuration_parameters(time_scale = ts,width=1,height=1)

    
    channel_env.set_float_parameter("coin_proba", coin_proba)
    channel_env.set_float_parameter("obstacle_proba",obstacle_proba)

    channel_env.set_float_parameter("move_speed",move_speed)
    channel_env.set_float_parameter("turn_speed",turn_speed)
    channel_env.set_float_parameter("momentum",momentum)
    channel_env.set_float_parameter("decrease_reward_on_stay",decrease_reward_on_stay)
    channel_env.set_float_parameter("coin_visible",coin_visible)
    
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    return(env,behavior_name,channel_env)



    
def to_uint8(img):
    img = np.array(127.5*(img.cpu().detach()+1)).astype(np.uint8).transpose(1,2,0)
    return(img)


def to_uint8_grayscale(img):
    img = np.array(127.5*(img.cpu().detach()+1)).astype(np.uint8)
    return(img)


def to_tensor(obs):
    return(2*(torch.tensor(obs).squeeze(0) - 0.5))

def to_tensor_grayscale(obs):
    return((2*(torch.tensor(obs).squeeze(0) - 0.5)).mean(axis=0))


