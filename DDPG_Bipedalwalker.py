import gym
from elegantrl.agents import AgentPPO
from elegantrl.train.config import get_gym_env_args, Arguments
from elegantrl.train.run import *

gym.logger.set_level(40) # Block warning
get_gym_env_args(gym.make("BipedalWalker-v3"), if_print=False)
env_func = gym.make
env_args = {
    "env_num": 1,
    "env_name": "BipedalWalker-v3",
    "max_step": 1600,
    "state_dim": 24,
    "action_dim": 4,
    "if_discrete": False,
    "target_return": 300,
    "id": "BipedalWalker-v3",
}
args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)
args.target_step = args.max_step * 4
args.gamma = 0.98
args.eval_times = 2**4
train_and_evaluate(args)