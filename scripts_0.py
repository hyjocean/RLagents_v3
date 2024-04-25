import numpy as np 
import networkx as nx
import time 
import copy
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from mMAPF_env.multi_env import mMAPFEnv
from algo.a2c import ActorNet, CriticNet
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
# from model.feature_mode import GridCNN
from utils.utils import load_config
from utils.map2nxG import create_graph_from_map2, nx_generate_path, lmrp_generate_path
from utils.rl_tools import discount
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
EXPERIENCE_BUFFER_SIZE = 128
NUM_BUFFERS = 1
def cpp_generate_path(world_state, agents_pos, agents_goal):
    world_state[world_state>0] =0
    world_state[world_state<0] =1
    tmp_padded_paths = cpp_mstar.find_path(world_state, agents_pos, agents_goal,1, 10)
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
            
def train_a2c(env, actor_net, critic_net, config, epochs=int(1e6), actor_lr=0.001, critic_lr=0.01, imi_epoch = 1):
    
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=actor_lr)
    actor_scheduler = lr_scheduler.StepLR(actor_optimizer, step_size=100, gamma=0.1)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=critic_lr)

    episode_buffers, s1Values = [[] for _ in range(1)], [
            [] for _ in range(1)]
    config['env_map_shape'] = env.observation_space['maps'].shape
    config['env_goal_shape'] = env.observation_space['goal_vector'].shape
    config['env_action_shape'] = env.action_space.n
    total_steps, i_buf = 0, 0

    for epoch in range(epochs):
        state = env.reset()
        done = False
        log_probs = []
        values = []
        
        if epoch < imi_epoch or np.random.rand() < DEFALT_PROB_IMITATION:
            agents_list = env.agents
            agents_num = len(agents_list)
            agents_pos = [agent.position for agent in agents_list]
            agents_goal = [agent.goal for agent in agents_list]
            agents_name = [agents.id_label for agents in agents_list]
            world_state = env.env.world.state.copy()

            # padded_paths = cpp_generate_path(world_state, agents_pos, agents_goal)  # cppmastar find path
            # padded_paths = nx_generate_path(world_state, agents_pos, agents_goal) # nx_grpah find path
            padded_paths = lmrp_generate_path(world_state, agents_pos, agents_goal, agents_name, config['PATH']) # lmrp find path
            if padded_paths == 0: # only used when apply nx_generate_path
                continue
            
            
            # 现在padded_paths是一个新的二维列表，所有子列表长度一致
            # print(f'padded_paths:{padded_paths}')
            seq_len = len(padded_paths) - 1
            result = env.env.parse_path(padded_paths, config['PATH'])
            if not result:
                continue
            obses, optimal_actions = {'maps':[],'goal_vector':[]}, []
            
            for agents in result:
                for obs, optimal_action in agents:
                    obses['maps'].append(obs['maps'])
                    obses['goal_vector'].append(obs['goal_vector'])
                    optimal_actions.append(optimal_action)
            # obses = {'maps':[0,0,0,0,0],'goal_vector':[0,0,0,0,0]}
            # optimal_actions = [0,0,0,0,0]
            obses['maps'] = np.array(obses['maps']).reshape((agents_num,seq_len,*(env.observation_space['maps'].shape)))
            obses['goal_vector'] = np.array(obses['goal_vector']).reshape((agents_num,seq_len,*(env.observation_space['goal_vector'].shape)))
            optimal_actions = np.array(optimal_actions).reshape((agents_num,seq_len))

            
            for batch in range(0, len(obses['maps']), 4):
                actor_optimizer.zero_grad()

                # rnn_state0 = actor_net.get_initial_state((1, agents_num, 512))
                pre_action_l, v_l, st_o, blk, ong, policy_sig = actor_net(obses['maps'][batch:batch+4], obses['goal_vector'][batch:batch+4], config)
                actor_loss = F.cross_entropy(pre_action_l.reshape(-1, config['env_action_shape']), torch.tensor(optimal_actions[batch:batch+4]).reshape(-1).to(pre_action_l.device))
                actor_loss.backward()
                actor_optimizer.step()

                if epoch % config['LOG']['print_interval'] == 0:
                    print(f"Epoch: {epoch}, episode: {batch}, imi loss: {actor_loss.detach().cpu().numpy()}")
                    wandb.log({"imi_loss": actor_loss.detach().item()})
            actor_scheduler.step()

        else:   
            rnn_state0 = actor_net.get_initial_state((env.num_agents, 512))
            s = [env.env._observe(agent.id_label) for agent in env.agents]
            episode_buffer, episode_values = [], []
            validActions = [env.env._listNextValidActions(agent.id_label-1)  for agent in env.agents]
            episode_reward = np.zeros(shape = (1,env.num_agents))
            episode_step_count = 0
            blocking = [False]*env.num_agents
            on_goal = [env.env.world.goals[tuple(agent.position)] == agent.id_label for agent in env.agents]
            wrong_block = np.array([0]*env.num_agents)
            wrong_on_goal = np.array([0]*env.num_agents)
            while not done:
                # state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                # maps, goal_vector = [],[]
                # for key in state.keys():
                #     maps.append(state[key]['maps'])
                #     goal_vector.append(state[key]['goal_vector'])
                # maps = np.stack(maps)
                # goal_vectors = np.stack(goal_vector)
                # a_probs, v, rnn_state, pred_blocking, pred_on_goal = actor_net(maps, goal_vectors)
                #[np.array(rnn_state0)[:int(key[-1])],np.array(rnn_state1)[:int(key[-1])]]
                
                maps = torch.tensor(np.array([state[key]['maps'] for key in state.keys()])).unsqueeze(1)
                goal_vectors = torch.tensor(np.array([state[key]['goal_vector'] for key in state.keys()])).unsqueeze(1)
                
                with torch.no_grad():
                    a_dist, v_l, st_o, blk, ong, policy_sig = actor_net(maps, goal_vectors, config, rnn_state0)

                train_valid = np.zeros_like(a_dist.detach().cpu().numpy())
                for i, valid_a in enumerate(validActions):
                    train_valid[i][0][valid_a] = 1
                # train_valid = np.array(train_valid)
                valid_dist = (a_dist.detach().cpu() * torch.tensor(train_valid))
                valid_dist /= valid_dist.sum(2)[:,:, np.newaxis]
                
                a = np.apply_along_axis(lambda x: np.random.choice(range(len(x)), size=1, p=x), 2, valid_dist).flatten()
                action_dict = {}
                for i, action in enumerate(a):
                    while action not in validActions[i]:
                        a[i] = np.random.choice(range(len(valid_dist[i]), size=1, p=valid_dist[i]))
                    action_dict[f'agent_f{i+1}'] = a[i]
            
                train_val = np.ones((env.num_agents,1))

                step_obs, step_r, step_d, step_info  = env.step(action_dict, config['PATH'])
                # step_res = [env.step(agent.id_label, action) for (agent, action) in zip(env.agents, a)]
                r = list(step_r.values())
                on_goal = [step_info[agent_name]['on_goal'].item() for agent_name in step_info.keys()]
                blocks = [step_info[agent_name]['blocking'] for agent_name in step_info.keys()]


                s1 = [env.env._observe(agent.id_label) for agent in env.agents]
                validActions = [env.env._listNextValidActions(agent.id_label, pre_action) for agent, pre_action in zip(env.agents, a)]

                d = env.env.finished


                episode_buffer.append([[d['maps'] for d in s], a, r, s1, d, v_l, train_valid, ong, 
                        [int(x) for x in on_goal] , blk.detach().cpu(), [int(x) for x in blocks], [d['goal_vector'] for d in s], train_val])
                episode_values.append(v_l.detach().cpu().numpy())
                episode_reward += np.array(r)
                s = s1
                rnn_state0 = st_o
                total_steps += 1
                episode_step_count += 1
                if d == True:
                    done = d
                    print('\n{} Goodbye World. We did it!'.format(
                            episode_step_count), end='\n')

                if (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):
                    if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                        episode_buffers[i_buf] = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                    else:
                        episode_buffers[i_buf] = episode_buffer[:]

                    if d:
                        s1Values[i_buf] = 0
                    else:
                        s1Values[i_buf] = v_l.detach().cpu().numpy().flatten().tolist()
                     

                    observations = [buffer[0] for buffer in episode_buffers[i_buf]]
                    goals = [buffer[-2] for buffer in episode_buffers[i_buf]]
                    actions = [buffer[1] for buffer in episode_buffers[i_buf]]
                    rewards = [buffer[2] for buffer in episode_buffers[i_buf]]
                    values = [[buffer[5][i].item() for i in range(env.num_agents)] for buffer in episode_buffers[i_buf]]
                    valids = [buffer[6] for buffer in episode_buffers[i_buf]]
                    blockings = [buffer[-4] for buffer in episode_buffers[i_buf]]
                    on_goal = [buffer[8] for buffer in episode_buffers[i_buf]]
                    train_value = [buffer[-1] for buffer in episode_buffers[i_buf]]

                    reward_plus = rewards + s1Values
                    discounted_rewards = [discount(rewards_p, config['gamma'])[:-1] for rewards_p in np.stack(reward_plus).T]
                    value_plus = values + s1Values
                    advantages = [reward + config['gamma'] * value_p[1:] - value_p[:-1] for reward, value_p in zip(np.stack(rewards).T, np.stack(value_plus).T)]
                    advantages = [discount(advantage, config['gamma']) for advantage in advantages]


                    p_l, v_l, _, b_l, on_gl, valids_l = actor_net(np.stack(observations).transpose(1,0,2,3,4), np.stack(goals).transpose(1,0,2,3),config, rnn_state0)
                    
                    value_loss = torch.tensor(np.stack(train_value)).permute(1,0,2).to(config['device']) \
                                           * torch.square(torch.tensor(np.stack(discounted_rewards)).unsqueeze(-1).to(config['device']) - v_l)
                    value_loss = value_loss.sum(-1).sum(-1)
                    entropy = p_l * torch.log(torch.clamp(p_l, min=1e-10, max=1.0))
                    entropy = - entropy.mean(-1).mean(-1)
                    
                    action_onehot = np.eye(config['env_action_shape'])[np.stack(actions).T]
                    responsible_out = torch.sum(p_l*torch.tensor(action_onehot).to(config['device']), axis = 2)
                    policy_loss = torch.log(torch.clamp(responsible_out,min=1e-15, max=1.0)) * torch.tensor(np.stack(advantages)).to(device)
                    policy_loss = - policy_loss.mean(-1)
                    # policy_loss = - torch.sum(torch.log(torch.clamp(responsible_out,min=1e-15, max=1.0)) * advantages)

                    valid_loss = torch.log(torch.clamp(valids_l,1e-10,1.0)) \
                                             * torch.tensor(np.stack(valids)).squeeze(2).permute(1,0,2).to(device)\
                                                 + torch.log(torch.clamp(1-valids_l,1e-10,1.0)) \
                                                    * (1-torch.tensor(np.stack(valids)).squeeze(2).permute(1,0,2)).to(device)
                    valid_loss = - valid_loss.mean(-1).mean(-1)

                    blockings = torch.stack(blockings).squeeze(-1).permute(1,0,2)
                    blocking_loss = blockings.to(config['device'])*torch.log(torch.clamp(b_l,1e-10,1.0)) \
                                        + (1-blockings.to(config['device']))*torch.log(torch.clamp(1-b_l,1e-10,1.0))
                    blocking_loss = - blocking_loss.mean(-1).mean(-1)

                    on_goal = torch.tensor(np.stack(on_goal)).T
                    on_goal_loss = on_goal.unsqueeze(-1).to(device)*torch.log(torch.clamp(on_gl,1e-10,1.0)) \
                                        + (1-on_goal.unsqueeze(-1).to(device))*torch.log(torch.clamp(1-on_gl,1e-10,1.0))
                    on_goal_loss = - on_goal_loss.mean(-1).mean(-1)

                    
                    
                    loss = 0.5 * value_loss + policy_loss + 0.5*valid_loss \
                            - entropy * 0.01 +.5*blocking_loss
                    

                    actor_optimizer.zero_grad()
                    loss.mean().backward()
                    actor_optimizer.step()
                    actor_optimizer.zero_grad()

                    i_buf = (i_buf+1)%NUM_BUFFERS
                    rnn_state0 = st_o
                    episode_buffers[i_buf] = []

                    if epoch % config['LOG']['print_interval'] == 0:
                        print(f"Epoch: {epoch}, episode: {episode_step_count}, rl loss: {loss.detach().cpu().numpy()} \t mean: {loss.mean().item()}")
                        wandb.log({"rl_loss": loss.mean().item()})
                        wandb.log({"value_loss": value_loss.mean().item()})
                        wandb.log({"policy_loss": policy_loss.mean().item()})
                        wandb.log({"valid_loss": valid_loss.mean().item()})
                        wandb.log({"entropy": entropy.mean().item()})
                        wandb.log({"blocking_loss": blocking_loss.mean().item()})

                if episode_step_count >= config["max_episode_length"] or d:
                    if d:
                        print('\n{} Goodbye World. We did it!'.format(
                            episode_step_count), end='\n')
                        log_path = config['LOG']['log_path']+config['PATH']['path_id'] 
                        if log_path:
                            with open(log_path, 'a') as f:
                                
                                f.write(f"\n{episode_step_count} Goodbye World. We did it!", end='\n')
                                f.write(f"Epoch: {epoch}, Episode: {episode_step_count}, finished after {total_steps} timesteps\n")
                                f.write(f"\tworld map: {env.world.state}\n")
                                for agent in env.agents:
                                    f.write(f"\tagent_name: {agent.id_label}, \tagent_st_pos: {agent.start_pos}, \t agent_goal: {agent.goal}\n")
                                f.write("\n\n")
                            f.close()
                    else:
                        print('\n{} Episode finished after {} timesteps'.format(
                            episode_step_count, total_steps), end='\n')
                    break


                    # # - tf.reduce_sum(tf.log(tf.clip_by_value(self.valids,1e-10,1.0)) *\
                    # #             self.train_valid+tf.log(tf.clip_by_value(1-self.valids,1e-10,1.0)) * (1-self.train_valid))
                    # for i in range(len(validActions)):
                    #     tmp = a_probs[i, 0, validActions[i]]

                    # # valid_dist = np.array(torch.gather(a_probs, 2, torch.tensor(validActions).unsqueeze(1).to(device)).cpu().detach())
                    # valid_dist /= np.sum(valid_dist, 2)[:,:,np.newaxis]

                    # wrong_block[(pred_blocking.flatten().detach().cpu().numpy() < 0.5) == blocking] += 1
                    # wrong_on_goal[(pred_on_goal.flatten().detach().cpu().numpy() < 0.5) == on_goal] += 1

                    # a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
                    # actions_index = np.apply_along_axis(lambda x: np.random.choice(range(len(x)), p=x), 2, valid_dist)
                    # actions = [validActions[i][a] for i, a in enumerate(actions_index)]

                    # value = actor_net.value_function()
                    # action = np.random.choice(env.action_space.n, p=np.squeeze(a_probs[0].cpu().detach().numpy()))
                    # log_prob = torch.log(a_probs[0].squeeze(0)[action])
                    # next_state, reward, done, _ = env.step(action)
                    # log_probs.append(log_prob)
                    # values.append(value)
                    # rewards.append(reward)
                    # state = next_state
                    # # Calculate losses for Actor and Critic
                    # returns = []
                    # R = 0
                    # for r in rewards[::-1]:
                    #     R = r + 0.99 * R  # Discount factor
                    #     returns.insert(0, R)
                    # returns = torch.tensor(returns)
                    # returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # Normalize
                    # actor_loss = -sum([log_probs[i] * (returns[i] - values[i].item()) for i in range(len(rewards))])
                    # critic_loss = sum([(returns[i] - values[i])**2 for i in range(len(rewards))])
                    # actor_optimizer.zero_grad()
                    # actor_loss.backward()
                    # actor_optimizer.step()
                    # critic_optimizer.zero_grad()
                    # critic_loss.backward()
                    # critic_optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Actor Loss = {actor_loss.detach().item()}')



config = load_config('/home/bld/HK_RL/RLagents_v3/config.yml')
device = torch.device(f"cuda:{config['gpu_id']}") if torch.cuda.is_available() else torch.device("cpu") 
# device = torch.device("cpu")

config['device'] = device


current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
config['PATH']['path_id'] = f"path_{current_time}"

print(f"device: {device}")
# 创建网络和环境
env = mMAPFEnv({'map_name':'mMAPF'})
actor_net = ActorNet(env.observation_space, env.action_space, config).to(device)
critic_net = CriticNet().to(device)
# actor_net = ActorNet(env.observation_space, env.action_space, config)
# critic_net = CriticNet()
train_a2c(env, actor_net, critic_net, config)
# 训练
wandb.finish()