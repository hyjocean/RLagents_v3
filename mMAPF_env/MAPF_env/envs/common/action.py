import functools
import itertools
from typing import TYPE_CHECKING, Optional, Union, Tuple, Callable, List
from gymnasium import spaces
import numpy as np
# from .abstract import AbstractEnv

from ....MAPF_env import utils
from ....MAPF_env.utils import Vector
# from MAPF_env.agents.behavior import IDMAgent
# from ...agents.dynamics import BicycleAgent
from ...agents.kinematics import Agent
from ...agents.controller import MDPAgent

if TYPE_CHECKING:
    from .abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType(object):

    """A type of action specifies its definition space, and how actions are executed in the environment"""

    def __init__(self, env: "AbstractEnv", **kwargs) -> None:
        self.env = env
        self.__controlled_agent = None

    def space(self) -> spaces.Space:
        """The action space."""
        raise NotImplementedError

    @property
    def agent_class(self) -> Callable:
        """
        The class of a agent able to execute the action.

        Must return a subclass of :py:class:`highway_env.agent.kinematics.Agent`.
        """
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """
        Execute the action on the ego-agent.

        Most of the action mechanics are actually implemented in agent.act(action), where
        agent is an instance of the specified :py:class:`highway_env.envs.common.action.ActionType.agent_class`.
        Must some pre-processing can be applied to the action based on the ActionType configurations.

        :param action: the action to execute
        """
        raise NotImplementedError

    def get_available_actions(self):
        """
        For discrete action space, return the list of available actions.
        """
        raise NotImplementedError

    @property
    def controlled_agent(self):
        """The agent acted upon.

        If not set, the first controlled agent is used by default."""
        return self.__controlled_agent or self.env.agent

    @controlled_agent.setter
    def controlled_agent(self, agent):
        self.__controlled_agent = agent


class BaseAction(ActionType):
    ACTIONS_ALL = {
        0: 'IDLE',
        1: 'MOVE',
        2: 'TURN90',
        3: "TURN-90",
        4: 'FASTER',
        # 5: 'SLOWER',
        # 6: 'coushu1',
        # 7: 'coushu2',
        # 8: 'coushu3'
    }

    def __init__(self,
                 env: 'AbstractEnv',
                 target_speeds: Optional[Vector] = None,
                 **kwargs) -> None:
        """
        Create a discrete action space of meta-actions.

        :param env: the environment
        :param longitudinal: include longitudinal actions
        :param lateral: include lateral actions
        :param target_speeds: the list of speeds the agent is able to track
        """
        super().__init__(env)
        self.actions = self.ACTIONS_ALL
        self.target_speeds = np.array(
            target_speeds) if target_speeds is not None else Agent.DEFAULT_TARGET_SPEEDS
        if self.actions is None:
            raise ValueError(
                "No Action is included")
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(len(self.actions))

    @property
    def agent_class(self) -> Callable:
        return functools.partial(MDPAgent, target_speeds=self.target_speeds)

    def act(self, action: Union[int, np.ndarray]) -> None:
        self.controlled_agent.act(self.actions[int(action)])

class EnvAgentsAction(ActionType):
    def __init__(self, 
                 env: 'AbstractEnv', 
                 **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.env = env 
        self.num_agents = env.num_agents

    def space(self):
        return MultiAgentActionSpace([BaseAction(self.env
                                                 ).space() for _ in range(self.num_agents)])
    

    
class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]


def action_factory(env: 'AbstractEnv', config: dict) -> ActionType:
    if config["type"] == "EnvAction":
        return EnvAgentsAction(env, **config)
    # if config["type"] == "ContinuousAction":
    #     return ContinuousAction(env, **config)
    # if config["type"] == "DiscreteAction":
    #     return DiscreteAction(env, **config)
    # elif config["type"] == "DiscreteMetaAction":
    #     return DiscreteMetaAction(env, **config)
    # elif config["type"] == "MultiAgentAction":
    #     return MultiAgentAction(env, **config)
    else:
        raise ValueError("Unknown action type")

class ContinuousAction(ActionType):

    """
    An continuous action space for throttle and/or steering angle.

    If both throttle and steering are enabled, they are set in this order: [throttle, steering]

    The space intervals are always [-1, 1], but are mapped to throttle/steering intervals through configurations.
    """

    ACCELERATION_RANGE = (-5, 5.0)
    """Acceleration range: [-x, x], in m/s²."""

    STEERING_RANGE = (-np.pi / 4, np.pi / 4)
    """Steering angle range: [-x, x], in rad."""

    def __init__(self,
                 env: 'AbstractEnv',
                 acceleration_range: Optional[Tuple[float, float]] = None,
                 steering_range: Optional[Tuple[float, float]] = None,
                 speed_range: Optional[Tuple[float, float]] = None,
                 longitudinal: bool = True,
                 lateral: bool = True,
                 dynamical: bool = False,
                 clip: bool = True,
                 **kwargs) -> None:
        """
        Create a continuous action space.

        :param env: the environment
        :param acceleration_range: the range of acceleration values [m/s²]
        :param steering_range: the range of steering values [rad]
        :param speed_range: the range of reachable speeds [m/s]
        :param longitudinal: enable throttle control
        :param lateral: enable steering control
        :param dynamical: whether to simulate dynamics (i.e. friction) rather than kinematics
        :param clip: clip action to the defined range
        """
        super().__init__(env)
        self.acceleration_range = acceleration_range if acceleration_range else self.ACCELERATION_RANGE
        self.steering_range = steering_range if steering_range else self.STEERING_RANGE
        self.speed_range = speed_range
        self.lateral = lateral
        self.longitudinal = longitudinal
        if not self.lateral and not self.longitudinal:
            raise ValueError(
                "Either longitudinal and/or lateral control must be enabled")
        self.dynamical = dynamical
        self.clip = clip
        self.size = 2 if self.lateral and self.longitudinal else 1
        self.last_action = np.zeros(self.size)

    def space(self) -> spaces.Box:
        return spaces.Box(-1., 1., shape=(self.size,), dtype=np.float32)

    @property
    def agent_class(self) -> Callable:
        return Agent 

    def get_action(self, action: np.ndarray):
        if self.clip:
            action = np.clip(action, -1, 1)
        if self.speed_range:
            self.controlled_agent.MIN_SPEED, self.controlled_agent.MAX_SPEED = self.speed_range
        if self.longitudinal and self.lateral:
            return {
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": utils.lmap(action[1], [-1, 1], self.steering_range),
            }
        elif self.longitudinal:
            return {
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": 0,
            }
        elif self.lateral:
            return {
                "acceleration": 0,
                "steering": utils.lmap(action[0], [-1, 1], self.steering_range)
            }

    def act(self, action: np.ndarray) -> None:
        self.controlled_agent.act(self.get_action(action))
        self.last_action = action


class DiscreteAction(ContinuousAction):
    def __init__(self,
                 env: 'AbstractEnv',
                 acceleration_range: Optional[Tuple[float, float]] = None,
                 steering_range: Optional[Tuple[float, float]] = None,
                 longitudinal: bool = True,
                 lateral: bool = True,
                 dynamical: bool = False,
                 clip: bool = True,
                 actions_per_axis: int = 3,
                 **kwargs) -> None:
        super().__init__(env, acceleration_range=acceleration_range, steering_range=steering_range,
                         longitudinal=longitudinal, lateral=lateral, dynamical=dynamical, clip=clip)
        self.actions_per_axis = actions_per_axis

    def space(self) -> spaces.Discrete:
        return spaces.Discrete(self.actions_per_axis**self.size)

    def act(self, action: int) -> None:
        cont_space = super().space()
        axes = np.linspace(cont_space.low, cont_space.high,
                           self.actions_per_axis).T
        all_actions = list(itertools.product(*axes))
        super().act(all_actions[action])


class DiscreteMetaAction(ActionType):

    """
    An discrete action space of meta-actions: lane changes, and cruise control set-point.
    """

    ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    """A mapping of action indexes to labels."""

    ACTIONS_LONGI = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    """A mapping of longitudinal action indexes to labels."""

    ACTIONS_LAT = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT'
    }
    """A mapping of lateral action indexes to labels."""

    def __init__(self,
                 env: 'AbstractEnv',
                 longitudinal: bool = True,
                 lateral: bool = True,
                 target_speeds: Optional[Vector] = None,
                 **kwargs) -> None:
        """
        Create a discrete action space of meta-actions.

        :param env: the environment
        :param longitudinal: include longitudinal actions
        :param lateral: include lateral actions
        :param target_speeds: the list of speeds the agent is able to track
        """
        super().__init__(env)
        self.longitudinal = longitudinal
        self.lateral = lateral
        self.target_speeds = np.array(
            target_speeds) if target_speeds is not None else MDPAgent.DEFAULT_TARGET_SPEEDS
        self.actions = self.ACTIONS_ALL if longitudinal and lateral \
            else self.ACTIONS_LONGI if longitudinal \
            else self.ACTIONS_LAT if lateral \
            else None
        if self.actions is None:
            raise ValueError(
                "At least longitudinal or lateral actions must be included")
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(len(self.actions))

    @property
    def agent_class(self) -> Callable:
        return functools.partial(MDPAgent, target_speeds=self.target_speeds)

    def act(self, action: Union[int, np.ndarray]) -> None:
        self.controlled_agent.act(self.actions[int(action)])

    def get_available_actions(self) -> List[int]:
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        actions = [self.actions_indexes['IDLE']]
        network = self.controlled_agent.road.network
        for l_index in network.side_lanes(self.controlled_agent.lane_index):
            if l_index[2] < self.controlled_agent.lane_index[2] \
                    and network.get_lane(l_index).is_reachable_from(self.controlled_agent.position) \
                    and self.lateral:
                actions.append(self.actions_indexes['LANE_LEFT'])
            if l_index[2] > self.controlled_agent.lane_index[2] \
                    and network.get_lane(l_index).is_reachable_from(self.controlled_agent.position) \
                    and self.lateral:
                actions.append(self.actions_indexes['LANE_RIGHT'])
        if self.controlled_agent.speed_index < self.controlled_agent.target_speeds.size - 1 and self.longitudinal:
            actions.append(self.actions_indexes['FASTER'])
        if self.controlled_agent.speed_index > 0 and self.longitudinal:
            actions.append(self.actions_indexes['SLOWER'])
        return actions



# class MultiAgentAction(ActionType):
#     def __init__(self,
#                  env: 'AbstractEnv',
#                  action_config: dict,
#                  **kwargs) -> None:
#         super().__init__(env)
#         self.action_config = action_config
#         self.agents_action_types = []
#         for agent in self.env.controlled_agents:
#             action_type = action_factory(self.env, self.action_config)
#             action_type.controlled_agent = agent
#             self.agents_action_types.append(action_type)

#     def space(self) -> spaces.Space:
#         return spaces.Tuple([action_type.space() for action_type in self.agents_action_types])

#     @property
#     def agent_class(self) -> Callable:
#         return action_factory(self.env, self.action_config).agent_class

#     def act(self, action: Action) -> None:
#         assert isinstance(action, tuple)
#         for agent_action, action_type in zip(action, self.agents_action_types):
#             action_type.act(agent_action)

#     def get_available_actions(self):
#         return itertools.product(*[action_type.get_available_actions() for action_type in self.agents_action_types])


#     def get_available_actions(self) -> List[int]:
#         """
#         Get the list of currently available actions.

#         Lane changes are not available on the boundary of the road, and speed changes are not available at
#         maximal or minimal speed.

#         :return: the list of available actions
#         """
#         actions = [self.actions_indexes['IDLE']]
#         network = self.controlled_agent.road.network
#         for l_index in network.side_lanes(self.controlled_agent.lane_index):
#             if l_index[2] < self.controlled_agent.lane_index[2] \
#                     and network.get_lane(l_index).is_reachable_from(self.controlled_agent.position) \
#                     and self.lateral:
#                 actions.append(self.actions_indexes['LANE_LEFT'])
#             if l_index[2] > self.controlled_agent.lane_index[2] \
#                     and network.get_lane(l_index).is_reachable_from(self.controlled_agent.position) \
#                     and self.lateral:
#                 actions.append(self.actions_indexes['LANE_RIGHT'])
#         if self.controlled_agent.speed_index < self.controlled_agent.target_speeds.size - 1 and self.longitudinal:
#             actions.append(self.actions_indexes['FASTER'])
#         if self.controlled_agent.speed_index > 0 and self.longitudinal:
#             actions.append(self.actions_indexes['SLOWER'])
        # return actions
