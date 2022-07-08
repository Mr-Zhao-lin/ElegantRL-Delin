import os
import time
import torch
import numpy as np
#训练
from elegantrl_helloworld.env import build_env
from elegantrl_helloworld.agent import ReplayBuffer, ReplayBufferList
import cv2

def train_agent(args):
    args.init_before_training()  #开点线程之类的
    gpu_id = args.learner_gpus

    '''init'''
    env = build_env(args.env_func, args.env_args)  #此处真正建立 gym环境
    #env.render()
    #print("ALL RIGHT")
    agent = args.agent_class(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)   #config 里面配置的elegantrl_helloworld.agent  DQN
    agent.states = [env.reset(), ]

    if args.if_off_policy:#如果off
        buffer = ReplayBuffer(gpu_id=gpu_id,
                              max_capacity=args.max_capacity,
                              state_dim=args.state_dim,
                              action_dim=1 if args.if_discrete else args.action_dim, )
        trajectory = agent.explore_env(env, args.target_step * args.eval_gap)
        buffer.update_buffer((trajectory,))
    else:
        buffer = ReplayBufferList()

    '''start training'''
    cwd = args.cwd
    eval_gap = args.eval_gap
    break_step = args.break_step
    target_step = args.target_step
    del args

    print("\n| `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`."
          "\n| `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action."
          "\n| `ObjC` denotes the objective of Critic network. Or call it loss function of critic network."
          "\n| `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network."
          "\n")

    start_time = time.time()
    total_step = 0
    eval_timer = start_time - eval_gap
    torch.set_grad_enabled(False)
    while True:
        trajectory = agent.explore_env(env, target_step)
        steps, r_exp = buffer.update_buffer((trajectory,))

        with torch.enable_grad():
            logging_tuple = agent.update_net(buffer)

        total_step += steps

        if eval_timer + eval_gap < time.time():
            eval_timer = time.time()
            print(f"| Steps {total_step:8.2e}  ExpR {r_exp:8.2f}  "
                  f"| ObjC {logging_tuple[0]:8.2f}  ObjA {logging_tuple[1]:8.2f}")
            save_path = f"{cwd}/actor_{total_step:012.0f}_{time.time() - start_time:08.0f}_{r_exp:08.2f}.pth"
            torch.save(agent.act.state_dict(), save_path)

        if (total_step > break_step) or os.path.exists(f"{cwd}/stop"):

            break  # stop training when reach `break_step` or `mkdir cwd/stop`

    print(f'| UsedTime: {time.time() - start_time:.0f} | SavedDir: {cwd}')

    state=env.reset()
    frame=env.render('rgb_array')
    image_info=frame.shape
    size=(image_info[1],image_info[0])
    print(size)
    fps=30
    save_dir="C:\My program\ElegantRL-master\elegantrl_helloworld\vedio"
   # os.chdir(r"C:\My program\ElegantRL-master\elegantrl_helloworld\vedio")
    vedio=cv2.VideoWriter('run.mp4',cv2.VideoWriter_fourcc('m','p','4','v'),fps,size)

    for i in range(1024):
        frame=env.render('rgb_array')
        vedio.write(frame)
        #cv2.imwrite(f'{save_dir}/{i:06}.png', frame)
        tensor_state=torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

        #tensor_action=agent.act.get_action(tensor_state.to(agent.device)).detach().cpu()
        print('tensor_state',tensor_state)
        tensor_action=agent.act.net(tensor_state).argmax(dim=1, keepdim=True)
        print('tensor_action',tensor_action)
        action=tensor_action[0, 0].numpy()
        print(action)
        state, reward, done, _=env.step(action)
        if done:
            state=env.reset()
            print("done")
    env.close()



def evaluate_agent(args):
    args.if_remove = False
    args.init_before_training()
    gpu_id = args.learner_gpus

    env = build_env(env_func=args.env_func, env_args=args.env_args)
    agent = args.agent_class(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    actor = agent.act

    cwd = args.cwd
    recorder_path = f"{cwd}/recorder.npy"

    recorder = list()

    torch.set_grad_enabled(False)
    model_names = [name for name in os.listdir(cwd) if name[:6] == 'actor_']
    model_names.sort()

    print("\n| `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`."
          "\n| `avgR` denotes average value of cumulative rewards, which is the sum of rewards in an episode."
          "\n| `stdR` denotes standard dev of cumulative rewards, which is the sum of rewards in an episode."
          "\n| `avgS` denotes the average number of steps in an episode."
          "\n")

    for model_name in model_names:
        model_path = f"{cwd}/{model_name}"
        actor.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

        rewards_steps_ary = [get_rewards_and_steps(env, actor) for _ in range(args.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        cumulative_rewards_avg = rewards_steps_ary[:, 0].mean()
        cumulative_rewards_std = rewards_steps_ary[:, 0].std()
        episode_steps_avg = rewards_steps_ary[:, 1].mean()

        name_split = model_name.split('_')
        steps = int(name_split[1])
        used_time = int(name_split[2])
        recorder.append((steps, used_time, cumulative_rewards_avg))
        print(f"| Steps {steps:8.2e}  "
              f"| avgR {cumulative_rewards_avg:9.3f}  stdR {cumulative_rewards_std:9.3f}  "
              f"| avgS {episode_steps_avg:6.0f}")
    recorder = np.array(recorder)
    np.save(recorder_path, recorder)

    import matplotlib as mpl
    mpl.use('Agg')  # write  before `import matplotlib.pyplot as plt`. `plt.savefig()` without a running X server
    import matplotlib.pyplot as plt
    x_axis = recorder[:, 0]
    y_axis = recorder[:, 2]
    plt.plot(x_axis, y_axis)
    plt.xlabel('#samples (Steps)')
    plt.ylabel('#Rewards (Score)')
    plt.grid()

    file_name = f"LearningCurve_{args.env_name}_{args.agent_class.__name__}.jpg"
    file_path = f"{cwd}/{file_name}"
    plt.title(file_name)
    plt.savefig(file_path)
    print(f"| Save learning curve in {file_path}")
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


def get_rewards_and_steps(env, act) -> (float, int):  # get cumulative_rewards and episode steps
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    cumulative_returns = 0.0  # sum of rewards in an episode
    episode_steps = 0
    for episode_steps in range(env.max_step):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        cumulative_returns += reward
        if done:
            break
    cumulative_returns = getattr(env, 'cumulative_returns', cumulative_returns)
    return cumulative_returns, episode_steps + 1



def train_and_evaluate(args):
    train_agent(args)
    evaluate_agent(args)
