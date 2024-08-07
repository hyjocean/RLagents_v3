import numpy as np 
import networkx as nx
import time 
import copy
import wandb
import argparse
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

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
from utils.utils import load_config, seed_set, configure_logger
from utils.map2nxG import create_graph_from_map2, nx_generate_path, lmrp_generate_path
from utils.rl_tools import discount

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from types import SimpleNamespace
from pathlib import Path
import logging





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
            
def train_a2c(env, actor_net, critic_net, config, epochs=int(1e4), actor_lr=0.001, critic_lr=0.01, imi_epoch = 10):

    logger = config['LOG']['logger']
    
    DEFALT_PROB_IMITATION = config['DEFALT_PROB_IMITATION']
    EXPERIENCE_BUFFER_SIZE = config['EXPERIENCE_BUFFER_SIZE']
    NUM_BUFFERS = 1
    
    actor_optimizer = create_optimizer_v2(actor_net, **optimizer_kwargs(SimpleNamespace(**config['optimal'])))
    actor_scheduler, num_epochs = create_scheduler(SimpleNamespace(**config['optimal']), actor_optimizer)
    # actor_optimizer = optim.Adam(actor_net.parameters(), lr=actor_lr)
    # actor_scheduler = lr_scheduler.StepLR(actor_optimizer, step_size=100, gamma=0.1)
    # critic_optimizer = optim.Adam(critic_net.parameters(), lr=critic_lr)

    episode_buffers, s1Values = [[] for _ in range(1)], [[] for _ in range(1)]
    total_steps, i_buf = 0, 0
    config['env_map_shape'] = env.observation_space['maps'].shape
    config['env_goal_shape'] = env.observation_space['goal_vector'].shape
    config['env_action_shape'] = env.action_space.n
    
    env.reset()
    reward_history = []
    record_table = np.zeros(shape=env.env.world.state.shape)
    for epoch in range(num_epochs):
        state = env.reset()
        log_probs = []
        values = []
        
        # if epoch < imi_epoch or np.random.rand() < DEFALT_PROB_IMITATION:
        #     agents_list = env.agents
        #     agents_num = len(agents_list)
        #     agents_pos = [agent.position for agent in agents_list]
        #     agents_goal = [agent.goal for agent in agents_list]
        #     agents_name = [agents.id_label for agents in agents_list]
        #     world_state = env.env.world.state.copy()
            
        #     # padded_paths = cpp_generate_path(world_state, agents_pos, agents_goal)  # cppmastar find path
        #     # padded_paths = nx_generate_path(world_state, agents_pos, agents_goal) # nx_grpah find path
        #     padded_paths = lmrp_generate_path(world_state, agents_pos, agents_goal, agents_name, config['PATH']) # lmrp find path
        #     if padded_paths == 0 or len(padded_paths) == 1: # 0 means no path, len(padded_paths) == 1 means init_agent_pos on it's goal.
        #         continue
            
            
        #     # 现在padded_paths是一个新的二维列表，所有子列表长度一致
        #     # print(f'padded_paths:{padded_paths}')
        #     seq_len = len(padded_paths) - 1
        #     result = env.env.parse_path(padded_paths, config['PATH'])
        #     if not result:
        #         continue
        #     obses, optimal_actions = {'maps':[],'goal_vector':[]}, []
            
        #     for agents in result:
        #         for obs, optimal_action in agents:
        #             obses['maps'].append(obs['maps'])
        #             obses['goal_vector'].append(obs['goal_vector'])
        #             optimal_actions.append(optimal_action)
        #     # obses = {'maps':[0,0,0,0,0],'goal_vector':[0,0,0,0,0]}
        #     # optimal_actions = [0,0,0,0,0]
        #     obses['maps'] = np.array(obses['maps']).reshape((agents_num,seq_len,*(env.observation_space['maps'].shape)))
        #     obses['goal_vector'] = np.array(obses['goal_vector']).reshape((agents_num,seq_len,*(env.observation_space['goal_vector'].shape)))
        #     optimal_actions = np.array(optimal_actions).reshape((agents_num,seq_len))

            
        #     batch_items = config['batch_items']
        #     for batch in range(0, len(obses['maps']), batch_items):
        #         actor_optimizer.zero_grad()

        #         # rnn_state0 = actor_net.get_initial_state((1, agents_num, 512))
        #         pre_action_l, v_l, st_o, blk, ong, policy_sig = actor_net(obses['maps'][batch:batch+batch_items], obses['goal_vector'][batch:batch+batch_items], config)
        #         actor_loss = F.cross_entropy(pre_action_l.reshape(-1, config['env_action_shape']), torch.tensor(optimal_actions[batch:batch+4]).reshape(-1).to(pre_action_l.device))
        #         actor_loss.backward()
        #         actor_optimizer.step()

        #         if epoch % config['LOG']['print_interval'] == 0:
        #             print(f"Epoch: {epoch}, episode: {batch}, imi loss: {actor_loss.detach().cpu().numpy()}")
        #             wandb.log({"imi_loss": actor_loss.detach().item()})
        #     actor_scheduler.step(epoch+1, 0)
        # new
        if epoch < imi_epoch or np.random.rand() < DEFALT_PROB_IMITATION:
            obses, optimal_actions = {'maps':[],'goal_vector':[]}, []
            sample_num = 5000
            for _ in range(sample_num):
                agents_list = env.agents
                
                shape = env.env.world.state.shape
                start_pos = np.random.randint(0,shape[0],2)
                while (start_pos == agents_list[0].goal).all() or env.env.world.state[start_pos[0]][start_pos[1]] != 0:
                    start_pos = np.random.randint(0,shape[0],2)
                env.env.world.state[agents_list[0].position[0]][agents_list[0].position[1]] = 0
                env.env.world.state[start_pos[0]][start_pos[1]] = 1
                agents_list[0].position = start_pos

                state = env.env._observe(agents_list[0].id_label)
                obses['maps'].append(state['maps'])
                obses['goal_vector'].append(state['goal_vector'])
                a_list = [0, 0, 0, 0, 0]
                if state['goal_vector'][0][0]>0:
                    a_list[2]+=1
                if state['goal_vector'][0][0]<0:
                    a_list[4]+=1
                if state['goal_vector'][0][1]>0:
                    a_list[1]+=1
                if state['goal_vector'][0][1]<0:
                    a_list[3]+=1
                # if state['goal_vector'][0][2] == 0:
                #     a_list[0]+=1

                optimal_actions.append(np.array(a_list)/sum(a_list))

            obses['maps'] = np.array(obses['maps']).reshape((sample_num,1,*(env.observation_space['maps'].shape)))
            obses['goal_vector'] = np.array(obses['goal_vector'])
            optimal_actions = np.array(optimal_actions)
            actor_optimizer.zero_grad()
            pre_action_l, v_l, st_o, blk, ong, policy_sig = actor_net(obses['maps'], obses['goal_vector'], config)
            actor_loss = F.cross_entropy(pre_action_l.reshape(-1, config['env_action_shape']), torch.tensor(optimal_actions).to(pre_action_l.device))
            actor_loss.backward()
            actor_optimizer.step()
            agents_num = len(agents_list)
            
            if epoch % config['LOG']['print_interval'] == 0:
                print(f"Epoch: {epoch}, imi loss: {actor_loss.detach().cpu().numpy()}")
                wandb.log({"imi_loss": actor_loss.detach().item()})
            actor_scheduler.step(epoch+1, 0)

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


            world_state = env.env.world.state.copy()
            agents_pos = [agent.position for agent in env.agents]
            agents_goal = [agent.goal for agent in env.agents]
            agents_name = [agents.id_label for agents in env.agents]
            padded_paths = lmrp_generate_path(world_state, agents_pos, agents_goal, agents_name, config['PATH']) 
            optimal_steps = len(padded_paths) - 1
            while not env.env.finished:
                
                pos_x,pos_y = env.agents[0].position
                record_table[pos_x][pos_y] += 1
                np.savetxt('records/my_array.csv', record_table, fmt='%d',delimiter=',')

                maps = torch.tensor(np.array([state[key]['maps'] for key in state.keys()])).unsqueeze(1)
                goal_vectors = torch.tensor(np.array([state[key]['goal_vector'] for key in state.keys()])).unsqueeze(1)
                
                with torch.no_grad():
                    a_dist, v_l, st_o, blk, ong, policy_sig = actor_net(maps, goal_vectors, config, rnn_state0)

                train_valid = np.zeros_like(a_dist.detach().cpu().numpy())
                for i, valid_a in enumerate(validActions):
                    train_valid[i][0][valid_a] = 1
                valid_dist = (a_dist.detach().cpu() * torch.tensor(train_valid))
                valid_dist /= valid_dist.sum(2)[:,:, np.newaxis]
                
                a = np.apply_along_axis(lambda x: np.random.choice(range(len(x)), size=1, p=x), 2, valid_dist).flatten()
                if episode_buffer == []:
                    pre_action = a.copy()
                action_dict = {}
                
                for i, action in enumerate(a):
                    while action not in validActions[i]:
                        a[i] = np.random.choice(range(len(valid_dist[i]), size=1, p=valid_dist[i]))
                    if a[i] != 0:
                        pre_action[i] = a[i]
                    action_dict[f'agent_f{i+1}'] = a[i]

                train_val = np.ones((env.num_agents,1))

                step_obs, step_r, step_d, step_info  = env.step(action_dict, config['PATH'])
                # step_res = [env.step(agent.id_label, action) for (agent, action) in zip(env.agents, a)]
                r = list(step_r.values())
                on_goal = [step_info[agent_name]['on_goal'].item() for agent_name in step_info.keys()]
                blocks = [step_info[agent_name]['blocking'] for agent_name in step_info.keys()]


                s1 = [env.env._observe(agent.id_label) for agent in env.agents] # step_obs['agent_fx']['maps'] == s1[x]['maps']
                validActions = [env.env._listNextValidActions(agent.id_label, pre_action) for agent, pre_action in zip(env.agents, a)]
                # validActions = [env.env._listNextValidActions(agent.id_label, pre_action) for agent, pre_action in zip(env.agents, pre_action)]

                # d = env.env.finished

                
                episode_buffer.append([[d['maps'] for d in s], a, r, s1, step_d, v_l, train_valid, ong, 
                        [int(x) for x in on_goal] , blk.detach().cpu(), [int(x) for x in blocks], [d['goal_vector'] for d in s], train_val])
                episode_values.append(v_l.detach().cpu().numpy())
                episode_reward += np.array(r)
                s = s1
                state = step_obs
                rnn_state0 = st_o
                total_steps += 1
                episode_step_count += 1

                if (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or step_d[0]):
                # if (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0):
                    if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                        episode_buffers[i_buf] = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                    else:
                        episode_buffers[i_buf] = episode_buffer[:]

                    if step_d[0]:
                        s1Values[i_buf] = [0]*env.num_agents
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
                    advantages = [reward + config['gamma'] * value_p1 - value_p2 for reward, value_p1, value_p2 in zip(np.stack(rewards).T, np.stack(value_plus)[1:].T, np.stack(value_plus)[:-1].T)]
                    # rewards = np.array(list(map(lambda x: x[0], rewards)))
                    # value_plus = np.array(list(map(lambda x: x[0], value_plus)))
                    # advantages = (rewards + 0.95*value_plus[1:] - value_plus[:-1]).reshape(1,-1)

                    advantages = [discount(advantage, config['gamma']) for advantage in advantages]

                    p_l, v_l, _, b_l, on_gl, valids_l = actor_net(np.stack(observations).transpose(1,0,2,3,4), np.stack(goals).transpose(1,0,2,3),config, rnn_state0)

                    value_loss = torch.tensor(np.stack(train_value)).permute(1,0,2).to(config['device']) \
                                           * torch.square(torch.tensor(np.stack(discounted_rewards)).unsqueeze(-1).to(config['device']) - v_l)
                    value_loss = value_loss.sum(-1).mean()

                    entropy = - p_l * torch.log(torch.clamp(p_l, min=1e-10, max=1.0))
                    entropy = entropy.sum(-1).mean(-1)
                    
                    action_onehot = np.eye(config['env_action_shape'])[np.stack(actions).T]
                    responsible_out = torch.sum(p_l*torch.tensor(action_onehot).to(config['device']), axis = 2)
                    policy_loss = - torch.log(torch.clamp(responsible_out,min=1e-15, max=1.0)) * ( torch.tensor(np.stack(advantages))).to(config['device'])
                    policy_loss = policy_loss.mean(-1)
                    # policy_loss = - torch.sum(torch.log(torch.clamp(responsible_out,min=1e-15, max=1.0)) * advantages)

                    valid_loss = torch.log(torch.clamp(valids_l,1e-10,1.0)) * torch.tensor(np.stack(valids)).squeeze(2).permute(1,0,2).to(config['device'])\
                                 + torch.log(torch.clamp(1-valids_l,1e-10,1.0)) * (1-torch.tensor(np.stack(valids)).squeeze(2).permute(1,0,2)).to(config['device'])
                    valid_loss = - valid_loss.sum(-1).mean(-1)

                    blockings = torch.stack(blockings).squeeze(-1).permute(1,0,2)
                    blocking_loss = blockings.to(config['device'])*torch.log(torch.clamp(b_l,1e-10,1.0)) \
                                        + (1-blockings.to(config['device']))*torch.log(torch.clamp(1-b_l,1e-10,1.0))
                    blocking_loss = - blocking_loss.sum(-1).mean(-1)

                    on_goal = torch.tensor(np.stack(on_goal)).T
                    on_goal_loss = on_goal.unsqueeze(-1).to(config['device'])*torch.log(torch.clamp(on_gl,1e-10,1.0)) \
                                        + (1-on_goal.unsqueeze(-1).to(config['device']))*torch.log(torch.clamp(1-on_gl,1e-10,1.0))
                    on_goal_loss = - on_goal_loss.sum(-1).mean(-1)

                    if (epoch+1) % 100 == 0:
                        config['entr_alpha'] = config['entr_alpha'] / 2
                    # print('policy_loss:', policy_loss.mean().item(), 'value_loss:', value_loss.mean().item(), 'valid_loss:', valid_loss.mean().item(), 'entropy:', entropy.mean().item(), 'blocking_loss:', blocking_loss.mean().item())
                    # loss =  0.5 * value_loss + policy_loss + 0.5*valid_loss \
                    #         - config['entr_alpha']*entropy  + 0.5*blocking_loss
                    loss = 0.1*value_loss + policy_loss - config['entr_alpha']*entropy
                    # loss = policy_loss - config['entr_alpha']*entropy

                    actor_optimizer.zero_grad()
                    loss.mean().backward()
                    actor_optimizer.step()
                    # actor_optimizer.zero_grad()

                    i_buf = (i_buf+1)%NUM_BUFFERS
                    rnn_state0 = st_o
                    episode_buffers[i_buf] = []

                    if epoch % config['LOG']['print_interval'] == 0:
                        # logger.info(f"Epoch: {epoch}, episode: {episode_step_count}, rl loss: {loss.detach().cpu().numpy()} \t mean: {loss.mean().item()} \t \
                        #        value_loss: {value_loss.mean().item()} \t policy_loss: {policy_loss.mean().item()} \t entropy: {entropy.mean().item()} \t")
                        logger.info(f"Epoch: {epoch}, \t episode: {episode_step_count}, \t rl loss: {loss.detach().cpu().numpy()} \t mean: {loss.mean().item():.4f} \t"
                                    f"value_loss: {value_loss.mean().item():.4f} \t policy_loss: {policy_loss.mean().item():.4f} \t entropy: {entropy.mean().item():.4f} \t optimal_steps: {optimal_steps}")
                        wandb.log({"rl_loss": loss.mean().item()})
                        wandb.log({"value_loss": value_loss.mean().item()})
                        wandb.log({"policy_loss": policy_loss.mean().item()})
                        wandb.log({"valid_loss": valid_loss.mean().item()})
                        wandb.log({"entropy": entropy.mean().item()})
                        wandb.log({"blocking_loss": blocking_loss.mean().item()})
                        wandb.log({"rewards": np.stack(rewards).mean()})
                        wandb.log({"discounted_rewards": np.stack(discounted_rewards).mean()})
                        wandb.log({"values": np.stack(values).mean()})
                        wandb.log({"advantages": np.stack(advantages).mean()})
                        

                
                if episode_step_count >= config["max_episode_length"] or step_d[0]:
                    
                    wandb.log({"episode_step_count / optimal_steps": episode_step_count / (optimal_steps+1e-8)})
                    wandb.log({"episode_reward": episode_reward, 'epoch': epoch})
                    # reward_history.append(episode_reward)
                    # if len(reward_history) % 100 == 0:
                    #     sma = np.convolve(reward_history, np.ones(50)/50, mode='valid')
                    #     wandb.log({"reward_sma": sma})
                    if step_d[0]:
                        log_path = config['LOG']['log_path']
                        logger.info(f"{episode_step_count} Goodbye World. We ({env.num_agents} agents) did it!\n")
                        if log_path:
                            logger.debug(f"{episode_step_count} Goodbye World. We ({env.num_agents} agents) did it!")
                            logger.debug(f"Epoch: {epoch}, take {episode_step_count} Episode_step to reach all goals. ")
                            logger.debug(f"world map: \n {env.env.world.state}\n")
                            for agent in env.agents:
                                logger.debug(f"\tagent_name: {agent.id_label}, \tagent_st_pos: {agent.start_pos}, \t agent_goal: {agent.goal}")
                                logger.debug(f"obsacle: {np.where(env.env.world.state == -1)}")
                            logger.debug(f"episode_step_count / optimal_steps: {episode_step_count / (optimal_steps+1e-8)}")
                            # logger.debug("\n\n")
                            if episode_step_count / (optimal_steps+1e-8) <= config["episode_step_count"]:
                                config["episode_step_count"] = episode_step_count / (optimal_steps+1e-8)
                                torch.save(actor_net, config['SAVE']['model'])
                                logger.debug(f"\n\n======Save model parameters with current min episode_step_count/optimal_steps {episode_step_count / (optimal_steps+1e-8)} in {config['SAVE']['model']}======\n\n")

                        env.env.finished = True
                    else:
                        # print(f"Current World Closed at {episode_step_count} episode_step. We failed. See Next Time!\n")
                        logger.info(f"Current World Closed at {episode_step_count} episode_step. We failed. See Next Time!\n")
                        break

def wandb_set(args,config):
    notes = f"map_size: {args.map_size}x{args.map_size}, map_density: {args.map_density}, imi_prob: {args.imi_prob}, seed: {config['SEED']}"
    mode = 'disabled' if not args.wandb_log else 'online'
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="RLagents_v3",
        notes=notes,
        # track hyperparameters and run metadata
        config=config,
        mode=mode,
    )


def config_loading(args, config_file):
    #config
    config = load_config(args.config_file)
    # device
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu") 
    config['device'] = device
    # seed
    seed = seed_set(config['SEED'])
    config['SEED'] = seed       

    # logname
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    file_path = Path(__file__).parent

    config['PATH']['path_id'] = f"path_{current_time}"
    config['LOG']['log_path'] = Path(f"{file_path}/logs/log_{current_time}.log")
    logger = configure_logger(config['LOG']['log_path'])
    # logger = logging.getLogger()
    config['LOG']['logger'] = logger
    logger.info(f"MODEL SEED: {config['SEED']}")
    logger.info(f"DEVICE: {device}")
    logger.info(f"CONFIG: {config}")

    config.update({'DEFALT_PROB_IMITATION': args.imi_prob})
    config['SAVE']['model'] = f"params/model_params_{args.map_size}_obstacle_{args.map_density}_imiprob_{args.imi_prob}_{config['SEED']}_tst4vis.pth"
    return config

def env_creator(args):
    return lambda config: mMAPFEnv({'map_name': 'mMAPF'}, args)

def main():
    parser = argparse.ArgumentParser(description='Train A2C on MAPF')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--map_size', type=int, default=40, help='map size')
    parser.add_argument('--map_density', type=float, default=0.01, help='obstacke density of map')
    parser.add_argument('--imi_prob', type=float, default=0.5, help='imitation probability')
    parser.add_argument('--wandb_log', action='store_true', help='log to wandb')
    parser.add_argument('--observation_size', type=int, default=10, help='the range of observation')
    # parser.add_argument('--render_mode', type=str, default='human', help='rgb_array or human')
    parser.add_argument('--render_mode', action='store_true', help='true for rgb_array or false for human')
    parser.add_argument('--config_file', type=str, default='/home/bld/HK_RL/RLagents_v3/config.yml', help='config file')
    args = parser.parse_args()


    config = config_loading(args, args.config_file)
    if args.render_mode:
        args.render_mode = 'rgb_array'
    else:
        args.render_mode = 'human'
    # 创建网络和环境
    # env = mMAPFEnv({'map_name':'mMAPF'}, args)
    # config1 = config.copy()
    # config1.update(env.env.config)
    # wandb_set(args,config1)
    
    # register_env('mMAPF-v0', env_creater(args))
    register_env('mMAPF-v1', env_creator(args))
    ray.init()
    config_ray = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment("mMAPF-v1")  # 环境名称
        .rollouts(num_rollout_workers=2)  # 训练时worker数量
        .framework("torch")  # 使用的深度学习框架
        .training(model={"fcnet_hiddens": [64, 64]})  # 神经网络隐藏层节点数
        .evaluation(evaluation_num_workers=1)  # evaluation worker数量
    )

    algo = config_ray.build()
    for _ in range(10):
        result = algo.train()
        print("episode mean reward: ", result["episode_reward_mean"] )  # 3. train it,

    algo.evaluate()  # 4. and evaluate it.
    
    


    # actor_net = ActorNet(env.observation_space, env.action_space, config).to(config['device'])
    # critic_net = CriticNet().to(config['device'])
    # actor_net = ActorNet(env.observation_space, env.action_space, config)
    # critic_net = CriticNet()
    # train_a2c(env, actor_net, critic_net, config)
    # 训练
    wandb.finish()

if __name__ == '__main__':
    main()