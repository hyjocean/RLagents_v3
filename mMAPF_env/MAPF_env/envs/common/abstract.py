from .graphics import EnvViewer
from .finite_mdp import finite_mdp
from .observation import observation_factory, ObservationType, MultiAgentObservationSpace
from .action import action_factory, Action, ActionType, MultiAgentActionSpace
from ...world.world import State, setWorld
import copy
import os

import yaml
import sys
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text
from pathlib import Path
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import RecordVideo
from gymnasium.utils import seeding
import numpy as np
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from ray.tune.utils import merge_dicts


from ...agents.kinematics import Agent
from utils.utils import load_config
Observation = TypeVar("Observation")


class AbstractEnv(gym.Env):

    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    speed. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    """
    observation_type: ObservationType
    action_type: ActionType
    _record_video_wrapper: Optional[RecordVideo]
    metadata = {
        'render_modes': ['human', 'rgb_array'],
    }

    PERCEPTION_DISTANCE = 5.0 * Agent.MAX_SPEED
    """The maximum distance of any vehicle present in the observation [m]"""

    def __init__(self, config: dict=None, world0=None, goals0=None, render_mode="human", blank_world=False):
        """
        Args:
            DIAGONAL_MOVEMENT: if the agents are allowed to move diagonally
            SIZE: size of a side of the square grid
            PROB: range of probabilities that a given block is an obstacle
            FULL_HELP
        """

        # Configuration
        self.config = self.default_config()
        self.configure(config)

        # Initialize member variables
        self.num_agents        = self.config['num_agents'] 
        #a way of doing joint rewards
        self.individual_rewards           = [0 for i in range(self.num_agents)]
        self.observation_size  = self.config['observation_size'] 
        self.SIZE              = self.config["SIZE"] if type(self.config["SIZE"]) is tuple else eval(self.config["SIZE"])
        self.PROB              = self.config["PROB"] if type(self.config["PROB"]) is tuple else eval(self.config["PROB"])
        self.FULL_HELP         = self.config["FULL_HELP"]
        self.finished          = self.config["finished"]
        self.DIAGONAL_MOVEMENT = self.config["DIAGONAL_MOVEMENT"]

        # Initialize data structures
        self.world = setWorld(self.config, world0, goals0, blank_world=blank_world)
        self.agents = self.world.agents

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()
        self.finished          = False
    
        # if DIAGONAL_MOVEMENT:
        #     self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(9)])
        # else:
        #     self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(5)])

        # Rendering
        self.viewer = None
        self._record_video_wrapper = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.enable_auto_render = False

        # num_agents=1, observation_size=10, DIAGONAL_MOVEMENT=False, SIZE=(10,40), PROB=(0,.5), FULL_HELP=False,blank_world=False):
    #     self.reset()
    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        config = load_config(Path(__file__).parent / 'default_config.yaml')
        config['offscreen_rendering'] = eval(config['offscreen_rendering'])
        return config

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    # @property
    # def agent(self) -> Agent:
    #     """First (default) controlled vehicle."""
    #     return self.controlled_agents[0] if self.controlled_agents else None

    # @agent.setter
    # def agent(self, agent: Agent) -> None:
    #     """Set a unique controlled vehicle."""
    #     self.controlled_agents = [agent]

    
    def update_metadata(self, video_real_time_ratio=2):
        frames_freq = self.config["simulation_frequency"] \
            if self._record_video_wrapper else self.config["policy_frequency"]
        self.metadata['render_fps'] = video_real_time_ratio * frames_freq


    def define_spaces(self) -> None:
        """
        Set the spaces in one env.
        """
        # self.observation_space = MultiAgentObservationSpace([agent.observation_space for agent in self.agents])
        # self.action_space = MultiAgentActionSpace([agent.action_space for agent in self.agents])

        self.observation_type = observation_factory(
            self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Returns a multi-objective vector of rewards.

        If implemented, this reward vector should be aggregated into a scalar in _reward().
        This vector value should only be returned inside the info dict.

        :param action: the last action performed
        :return: a dict of {'reward_name': reward_value}
        """
        raise NotImplementedError

    def _is_terminated(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        raise NotImplementedError

    def _is_truncated(self) -> bool:
        """
        Check we truncate the episode at the current step

        :return: is the episode truncated
        """
        raise NotImplementedError

    def _info(self, obs: Observation, action: Optional[List[Action]] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        
        info = {
            "speed": [agent.speed for agent in self.world.agents],
            "crashed": [agent.crashed for agent in self.world.agents]
            # "action": [action,
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              ) -> Tuple[Observation, dict]:
        """
        Reset the environment to it's initial configuration

        :param seed: The seed that is used to initialize the environment's PRNG
        :param options: Allows the environment configuration to specified through `options["config"]`
        :return: the observation of the reset state
        """
        super().reset(seed=seed, options=options)
        if options and "config" in options:
            self.configure(options["config"])
        self.viewer = None
        self.update_metadata()
        # First, to set the controlled vehicle class depending on action space
        self.define_spaces()
        self.time = self.steps = 0
        self.done = False
        self._reset()
        # Second, to link the obs and actions to the vehicles once the scene is created
        self.define_spaces()
        
        # obs = self.observation_type.observe() #TODO: obs应该是一个长为n_agent的list，每个元素为每个agent的观测
        # info = self._info(obs, action=self.action_space.sample())
        if self.render_mode == 'human':
            self.render()
        # return obs, info

    def _reset(self) -> None:
        """
        Reset the scene: roads and vehicles.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()

    def step(self, action, episode=0):
        obs, reward, done, nextActions, on_goal, blocking, valid_action = self._step(action, episode)
        info = {"done": done, 
                "nextActions":nextActions, 
                "on_goal": on_goal, 
                "blocking": blocking,
                "valid_action": valid_action
                }
        if self.render_mode == 'human':
            self.render()
        return obs, reward, done, info


    def _step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.world is None or self.world.agents is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] //
                     self.config["policy_frequency"])
        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            self.world.act()
            self.world.step(1 / self.config["simulation_frequency"])
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if self.render_mode == 'rgb_array':
            image = self.viewer.get_image()
            return image

    def close(self) -> None:
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def get_available_actions(self) -> List[int]:
        return self.action_type.get_available_actions()

    def set_record_video_wrapper(self, wrapper: RecordVideo):
        self._record_video_wrapper = wrapper
        self.update_metadata()

    def _automatic_rendering(self) -> None:
        """
        Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps corresponding to agent decision-making.
        If a RecordVideo wrapper has been set, use it to capture intermediate frames.
        """
        if self.viewer is not None and self.enable_auto_render:

            if self._record_video_wrapper and self._record_video_wrapper.video_recorder:
                self._record_video_wrapper.video_recorder.capture_frame()
            else:
                self.render()

    def simplify(self) -> 'AbstractEnv':
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.

        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE)

        return state_copy


    # def set_preferred_lane(self, preferred_lane: int = None) -> 'AbstractEnv':
    #     env_copy = copy.deepcopy(self)
    #     if preferred_lane:
    #         for v in env_copy.road.vehicles:
    #             if isinstance(v, IDMVehicle):
    #                 v.route = [(lane[0], lane[1], preferred_lane)
    #                            for lane in v.route]
    #                 # Vehicle with lane preference are also less cautious
    #                 v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
    #     return env_copy

    # def set_route_at_intersection(self, _to: str) -> 'AbstractEnv':
    #     env_copy = copy.deepcopy(self)
    #     for v in env_copy.road.vehicles:
    #         if isinstance(v, IDMVehicle):
    #             v.set_route_at_intersection(_to)
        return env_copy

    def set_vehicle_field(self, args: Tuple[str, object]) -> 'AbstractEnv':
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v is not self.vehicle:
                setattr(v, field, value)
        return env_copy

    def call_vehicle_method(self, args: Tuple[str, Tuple[object]]) -> 'AbstractEnv':
        method, method_args = args
        env_copy = copy.deepcopy(self)
        for i, v in enumerate(env_copy.road.vehicles):
            if hasattr(v, method):
                env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
        return env_copy

    # def randomize_behavior(self) -> 'AbstractEnv':
    #     env_copy = copy.deepcopy(self)
    #     for v in env_copy.road.vehicles:
    #         if isinstance(v, IDMVehicle):
    #             v.randomize_behavior()
    #     return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1/self.config["policy_frequency"])

    def __deepcopy__(self, memo):
        """Perform a deep copy but without copying the environment viewer."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', '_record_video_wrapper']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if self.render_mode == 'rgb_array':
            image = self.viewer.get_image()
            return image
        
        
class MultiAgentWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward = info["agents_rewards"]
        terminated = info["agents_terminated"]
        truncated = info["agents_truncated"]
        return obs, reward, terminated, truncated, info
