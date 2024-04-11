import numpy as np 
import networkx as nx
import time 
import copy
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mMAPF_env.multi_env import mMAPFEnv
from algo.a2c import ActorNet, CriticNet
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import threading
# max_threads = 1
# thread_pool = threading.BoundedSemaphore(value=max_threads)
# from memory_profiler import profile
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
# from model.feature_mode import GridCNN
from utils.utils import load_config
from utils.map2nxG import create_graph_from_map2, nx_generate_path
# import matplotlib.pyplot as plt
# import tracemalloc

# tracemalloc.start()

# # 打开交互模式
# plt.ion()
wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="RLagents_v3",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.01,
    "architecture": "CNN",
    "dataset": "Random-map",
    "epochs": 1000,
    }
)


DEFALT_PROB_IMITATION = 0.5
def cpp_generate_path(world_state, agents_pos, agents_goal):
    world_state[world_state>0] =0
    world_state[world_state<0] =1
    tmp_padded_paths = cpp_mstar.find_path(world_state, agents_pos, agents_goal, 1, 5)
    padded_paths = []
    for i, path in enumerate(tmp_padded_paths):
        if i == 0:
            padded_paths.append(path)
            continue
        if path[0] == padded_paths[-1][1]:
            padded_paths.append([padded_paths[-1][0], path[1]])
        if path[1] == padded_paths[-1][0]:
            padded_paths.append([path[0], padded_paths[-1][1]])
        padded_paths.append(path)
    return padded_paths
            
def train_a2c(env, actor_net, critic_net, epochs=1000, actor_lr=0.01, critic_lr=0.01, imi_epoch = 100):
    
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=critic_lr)

    for epoch in range(epochs):
        state = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        if epoch < imi_epoch or np.random.rand() < DEFALT_PROB_IMITATION:
            agents_list = env.agents
            agents_num = len(agents_list)
            agents_pos = [agent.position for agent in agents_list]
            agents_goal = [agent.goal for agent in agents_list]
            world_state = env.env.world.state.copy()

            padded_paths = cpp_generate_path(world_state, agents_pos, agents_goal)  # cppmastar find path
            # padded_paths = nx_generate_path(world_state, agents_pos, agents_goal) # nx_grpah find path
            if padded_paths == 0: # only used when apply nx_generate_path
                continue
            
            
            # 现在padded_paths是一个新的二维列表，所有子列表长度一致
            print(f'padded_paths:{padded_paths}')
            result = env.env.parse_path(padded_paths)
            obses, optimal_actions = {'maps':[0]*(len(padded_paths)-1)*agents_num,'goal_vector':[0]*(len(padded_paths)-1)*agents_num}, [0]*(len(padded_paths)-1)*agents_num
            i = 0
            for agents in result:
                for obs, optimal_action in agents:
                    obses['maps'][i] = obs['maps']
                    obses['goal_vector'][i]= obs['goal_vector']
                    optimal_actions[i]=optimal_action
                    i+=1
            del i
            for batch in range(0, len(obses['maps']), 4):
                actor_optimizer.zero_grad()
                pre_action = actor_net(obses['maps'][batch:batch+4], obses['goal_vector'][batch:batch+4])
                actor_loss = F.cross_entropy(pre_action[0],torch.tensor(optimal_actions[batch:batch+4]).to(pre_action[0].device))
                actor_loss.backward()
                actor_optimizer.step()
                wandb.log({"actor_loss": actor_loss.detach().item()})

        else:   
            while not done:
                # state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                probs = actor_net(state['agent_f1'])
                value = actor_net.value_function()
                action = np.random.choice(env.action_space.n, p=np.squeeze(probs[0].cpu().detach().numpy()))
                log_prob = torch.log(probs[0].squeeze(0)[action])
                next_state, reward, done, _ = env.step(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                state = next_state
                # Calculate losses for Actor and Critic
                returns = []
                R = 0
                for r in rewards[::-1]:
                    R = r + 0.99 * R  # Discount factor
                    returns.insert(0, R)
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # Normalize
                actor_loss = -sum([log_probs[i] * (returns[i] - values[i].item()) for i in range(len(rewards))])
                critic_loss = sum([(returns[i] - values[i])**2 for i in range(len(rewards))])
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Actor Loss = {actor_loss.detach().item()}')


config = load_config('/home/bld/HK_RL/RLagents_v3/config.yml')
# device = torch.device(f"cuda:{config['gpu_id']}") if torch.cuda.is_available() else torch.device("cpu") 
device = torch.device("cpu")
config['device'] = device
print(f"device: {device}")
# 创建网络和环境
env = mMAPFEnv({'map_name':'mMAPF'})
actor_net = ActorNet(env.observation_space, env.action_space, config).to(device)
critic_net = CriticNet().to(device)
# actor_net = ActorNet(env.observation_space, env.action_space, config)
# critic_net = CriticNet()
train_a2c(env, actor_net, critic_net)
# for i in range(10):
#     thread = threading.Thread(target=train_a2c, args=(env, actor_net, critic_net))
#     thread_pool.acquire()
#     thread.start()

# for i in range(10):
#     thread_pool.release()
#     thread.join()
# 训练
wandb.finish()