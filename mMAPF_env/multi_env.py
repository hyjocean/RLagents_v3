import gymnasium as gym
from mMAPF_env.MAPF_env.envs.baseenv import MAPFEnv
from ray.tune.utils import merge_dicts
from collections import OrderedDict
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Dict as GymDict, Box
# from marllib import marl
# from marllib.envs.base_env import ENV_REGISTRY
import time
# from od_mstar3 import cpp_mstar

# MAPF_env.register_mapf_envs()
# env = gym.make('mapf-v0')
REGISTRY = {}
REGISTRY["mMAPF"] = MAPFEnv

policy_mapping_dict = {
    "mMAPF": {
        "description": "multi agent path finding",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    }

class mMAPFEnv(MultiAgentEnv):

    # Initialize env
    def __init__(self, env_config, args):
        map = env_config["map_name"]
        env_config.pop("map_name", None)

        self.env = REGISTRY[map]({'SIZE': tuple((args.map_size,args.map_size)), 'PROB': tuple((0, args.map_density)), 'observation_size': args.observation_size}, args.render_mode)
        # assume all agent same action/obs space
        self.configure(self.env.config, args)
        self.action_space = self.env.action_space[0]
        self.observation_space = GymDict(OrderedDict(
                maps=Box(-np.inf, np.inf,
                                shape=self.env.observation_space[0]['maps'].shape, dtype=np.float64),
                goal_vector=Box(-np.inf, np.inf,
                                         shape=(1,3), dtype=np.float64),
            ))
        # self.observation_space = GymDict({"obs":  Box(
        #     low=0.0,
        #     high=1.0,
        #     shape=(self.env.observation_space[0].shape[0],),
        #     dtype=np.dtype("float64"))})
        # self.agents = ["agent_"]
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config
    
    def configure(self, config: dict, args) -> None:
        self.env.config.update({'SIZE': tuple((args.map_size, args.map_size)), 'PROB': tuple((0, args.map_density))})

    def reset(self):
        original_obs, original_info = self.env.reset()
        obs = {}
        for i, agent in enumerate(self.agents):
            name = f'agent_f{agent.id_label}'
            obs[name] = OrderedDict({"maps": np.array(original_obs[i]['maps']), "goal_vector": np.array(original_obs[i]['goal_vector'])})
        
        self.last_obs = obs
        return obs

    def step(self, action_dict, path_cfg):
        action_ls = [action_dict[key] for key in action_dict.keys()]

        # print("\n"+"=="*10+"\n")
        # print(self.env.world.state+10*self.env.world.goals)
        # print("\n"+"=="*10+"\n")
        o, r, d, info = self.env.step(action_ls, path_cfg)
        rewards = {}
        obs = {}
        infos = {}
        for i, key in enumerate(action_dict.keys()):
            rewards[key] = r[i]
            obs[key] = OrderedDict({"maps": np.array(o[i]['maps']), "goal_vector": np.array(o[i]['goal_vector'])})
            infos[key] = OrderedDict({'done':info['done'], 'nextActions': np.array(info['nextActions'][i]), 'on_goal': np.array(info['on_goal'][i]), 'blocking': info['blocking'][i], 'valid_action': info['valid_action'][i]})
            # if action_dict[key] not in self.goal_act(self.last_obs[key]['goal_vector']):
            #     rewards[key] -= 0.2
            # if obs[key]["goal_vector"][0][-1] > self.last_obs[key]['goal_vector'][0][-1]:
            #     rewards[key] -= 0.2
            # {
            #     "obs": np.array(o[i])
            # }
        # dones = {"__all__": True if sum(d) == self.num_agents else False}
        dones = {"__all__": info['done'][0]}
        
        # print(rewards)
        self.last_obs = obs
        return obs, rewards, info['done'], infos

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        self.env.render()
        time.sleep(0.05)
        return True

    @property
    def agents(self):
        return self.env.agents


    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 50,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
    
    def goal_act(self, goal_vector):
        # goal_vector: [x, y, dis]
        # 0: idle
        # 1: right
        # 2: down
        # 3: left
        # 4: up
        dx,dy,mag = goal_vector[0]
        if dx > 0 and dy > 0:
            return [1,2]
        elif dx > 0 and dy == 0:
            return [2]
        elif dx > 0 and dy < 0:
            return [2,3]
        elif dx == 0 and dy > 0:
            return [1]
        elif dx == 0 and dy == 0:
            return [0]
        elif dx == 0 and dy < 0:
            return [3]
        elif dx < 0 and dy > 0:
            return [1,4]
        elif dx < 0 and dy == 0:
            return [4]
        elif dx < 0 and dy < 0:
            return [3,4]
# if __name__ == '__main__':
#     # register new env
#     ENV_REGISTRY["mMAPF"] = mMAPFEnv
#     # initialize env
#     env = marl.make_env(environment_name="mmapf", map_name="Checkers", abs_path="../../examples/config/env_config/magym.yaml")
#     # pick mappo algorithms
#     mappo = marl.algos.mappo(hyperparam_source="test")
#     # customize model
#     model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
#     # start learning
#     mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=True, num_gpus=1,
#               num_workers=2, share_policy='all', checkpoint_freq=50)