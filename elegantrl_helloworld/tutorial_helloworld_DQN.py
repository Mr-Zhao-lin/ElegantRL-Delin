# coding=utf-8
from elegantrl_helloworld.config import Arguments
from elegantrl_helloworld.run import train_and_evaluate
from elegantrl_helloworld.env import get_gym_env_args
import torch
import gym
import yaml
import cv2
gym.logger.set_level(40)  # Block warning

def train_dqn_in_cartpole(gpu_id=0):  # DQN is a simple but low sample efficiency.
    env_name = "CartPole-v0"
    alg = "DQN"
    with open("config.yml", 'rb') as f:
        hyp = yaml.safe_load(f)[alg][env_name] #导入环境和算法（agent）对应的参数
    env = gym.make(env_name)  #产生一个gym环境

    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)#获取gym参数

    args = Arguments(env_func, env_args, hyp) #

    #到目前为止，获得了一大堆参数  一个make方法可以用于产生环境，env_args是环境的参数，hyp是超惨
   
    train_and_evaluate(args)



def train_dqn_in_lunar_lander(gpu_id=0):  # DQN is a simple but low sample efficiency.
    env_name = "LunarLander-v2"
    alg = "DQN"
    with open("config.yml", 'r') as f:
        hyp = yaml.safe_load(f)[alg][env_name]
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(env_func, env_args, hyp)

    train_and_evaluate(args)



if __name__ == "__main__":

    train_dqn_in_cartpole()
    #train_dqn_in_lunar_lander()
