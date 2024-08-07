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
from utils.utils import load_config, configure_logger, seed_set
from utils.map2nxG import create_graph_from_map2, nx_generate_path, lmrp_generate_path
from utils.rl_tools import discount

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from types import SimpleNamespace
from pathlib import Path
import argparse

def test_a2c(env, actor, critic, config):
    actor.eval()
    episode_step_count = 0
    state = env.reset()
    agents_list = env.agents
    agents_pos = [agent.position for agent in agents_list]
    validActions = [env.env._listNextValidActions(agent.id_label-1)  for agent in env.agents]
    config['env_map_shape'] = env.observation_space['maps'].shape
    config['env_goal_shape'] = env.observation_space['goal_vector'].shape
    config['env_action_shape'] = env.action_space.n

    maps = torch.tensor(np.array([state[key]['maps'] for key in state.keys()])).unsqueeze(1)
    goal_vectors = torch.tensor(np.array([state[key]['goal_vector'] for key in state.keys()])).unsqueeze(1)
    rnn_state0 = actor.get_initial_state((env.num_agents, 512))


    world_state = env.env.world.state.copy()
    agents_pos = [agent.position for agent in env.agents]
    agents_goal = [agent.goal for agent in env.agents]
    agents_name = [agents.id_label for agents in env.agents]
    padded_paths = lmrp_generate_path(world_state, agents_pos, agents_goal, agents_name, config['PATH']) 
    optimal_steps = len(padded_paths) - 1
    while not env.env.finished:
        a_dist, v_l, st_o, blk, ong, policy_sig = actor(maps, goal_vectors, config, rnn_state0)

        train_valid = np.zeros_like(a_dist.detach().cpu().numpy())
        for i, valid_a in enumerate(validActions):
            train_valid[i][0][valid_a] = 1
        valid_dist = (a_dist.detach().cpu() * torch.tensor(train_valid))
        valid_dist /= valid_dist.sum(2)[:,:, np.newaxis]

        a = np.apply_along_axis(lambda x: np.random.choice(range(len(x)), size=1, p=x), 2, valid_dist).flatten()
        # critic.load_state_dict(torch.load(config['critic']))
        action_dict = {}
        for i, action in enumerate(a):
            while action not in validActions[i]:
                a[i] = np.random.choice(range(len(valid_dist[i]), size=1, p=valid_dist[i]))
            # if a[i] != 0:
            #     pre_action[i] = a[i]
            action_dict[f'agent_f{i+1}'] = a[i]

        step_obs, step_r, step_d, step_info  = env.step(action_dict, config['PATH'])
        # time.sleep(1)
        episode_step_count += 1
        state = step_obs
        validActions = [env.env._listNextValidActions(agent.id_label, pre_action) for agent, pre_action in zip(env.agents, a)]
        rnn_state0 = st_o

        if step_d[0]:
            print('Episode finished after {} timesteps'.format(episode_step_count))
            print(f'Optimal steps: {optimal_steps}')
            print(f"ratio: {episode_step_count/(optimal_steps+1e-8)}")
            return True
        elif episode_step_count > 500:
            print(f"No solution found in 500 steps")
            return False




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
    config['SAVE']['model'] = f"params/model_params_{args.map_size}_obstacle_{args.map_density}_imiprob_{args.imi_prob}_{config['SEED']}.pth"
    return config
    
def main():
    parser = argparse.ArgumentParser(description='Train A2C on MAPF')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--map_size', type=int, default=20, help='map size')
    parser.add_argument('--map_density', type=float, default=0.1, help='obstacke density of map')
    parser.add_argument('--imi_prob', type=float, default=0.2, help='imitation probability')
    parser.add_argument('--wandb_log', action='store_true', help='log to wandb')
    parser.add_argument('--observation_size', type=int, default=10, help='the range of observation')
    parser.add_argument('--render_mode', type=str, default='human', help='rgb_array or human')
    # parser.add_argument('--render_mode', action='store_true', help='true for rgb_array or false for human')
    parser.add_argument('--config_file', type=str, default='/home/bld/HK_RL/RLagents_v3/config.yml', help='config file')
    args = parser.parse_args()


    config = config_loading(args, args.config_file)
    # if args.render_mode:
    #     args.render_mode = 'rgb_array'
    # else:
    #     args.render_mode = 'human'
    # 创建网络和环境


    env = mMAPFEnv({'map_name':'mMAPF'}, args)


    # wandb_set(args,config.copy().update(env.env.config))
    actor_net = torch.load(f"params/model_params_20_obstacle_0.1_imiprob_0.2_1883.pth").to(config['device'])
    critic_net = CriticNet().to(config['device'])
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    config['PATH']['path_id'] = f"tst_path_{current_time}"

    # actor_net = ActorNet(env.observation_space, env.action_space, config)
    # critic_net = CriticNet()
    # test_a2c(env, actor_net, critic_net, config)

    success_times = 0
    for i in range(100):
        success_times += test_a2c(env, actor_net, critic_net, config)
        # if    test_a2c(env, actor_net, critic_net, config)
    print(f"success rate: {success_times/100}")

if __name__ == '__main__':
    main()

# actor_net = ActorNet(env.observation_space, env.action_space, config).to(device)



