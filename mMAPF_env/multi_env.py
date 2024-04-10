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

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)

        self.env = REGISTRY[map](**env_config)
        # assume all agent same action/obs space
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

    def reset(self):
        original_obs, original_info = self.env.reset()
        obs = {}
        for i, agent in enumerate(self.agents):
            name = f'agent_f{agent.id_label}'
            obs[name] = OrderedDict({"maps": np.array(original_obs[i]['maps']), "goal_vector": np.array(original_obs[i]['goal_vector'])})
        return obs

    def step(self, action_dict):
        action_ls = [action_dict[key] for key in action_dict.keys()]
        # print("\n"+"=="*10+"\n")
        # print(self.env.world.state+10*self.env.world.goals)
        # print("\n"+"=="*10+"\n")
        o, r, d, info = self.env.step(action_ls)
        rewards = {}
        obs = {}
        infos = {}
        for i, key in enumerate(action_dict.keys()):
            rewards[key] = r[i]
            obs[key] = OrderedDict({"maps": np.array(o[i]['maps']), "goal_vector": np.array(o[i]['goal_vector'])})
            infos[key] = OrderedDict({'done':info['done'], 'nextActions': np.array(info['nextActions'][i]), 'on_goal': np.array(info['on_goal'][i]), 'blocking': info['blocking'][i], 'valid_action': info['valid_action'][i]})
            # {
            #     "obs": np.array(o[i])
            # }
        # dones = {"__all__": True if sum(d) == self.num_agents else False}
        dones = {"__all__": info['done'][0]}
        # print(rewards)
        return obs, rewards, dones, infos

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