from typing import Dict, Text
# from gymnasium.envs.classic_control import rendering
import numpy as np
import time
import pygame
import math
import copy
import sys
from pathlib import Path
from ..world.world import State, setWorld
# from MAPF_env.world.world import State, setWorld
from ..envs.common.abstract import AbstractEnv
from ..envs.common.action import Action
from ..agents.kinematics import Agent
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
# from gymnasium.envs.registration import register
# import os
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import sys

from utils.utils import load_config
Observation = np.ndarray

ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD,FINISH_REWARD,BLOCKING_COST = -0.3, -.5, 5.0, -2.,20.,-1.
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
JOINT = False # True for joint estimation of rewards for closeby agents
dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
actionDict={v:k for k,v in dirDict.items()}


class MAPFEnv(AbstractEnv):

    def getFinishReward(self):
        return FINISH_REWARD
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(load_config(Path(__file__).parent / 'config.yaml'))
        config['SIZE'] = config['SIZE'] if type(config['SIZE']) is tuple else eval(config['SIZE'])
        config['PROB'] = config['PROB'] if type(config['PROB']) is tuple else eval(config['PROB'])
       
        return config
    # Initialize env
    
    def isConnected(self,world0):
        sys.setrecursionlimit(10000)
        world0 = world0.copy()

        def firstFree(world0):
            for x in range(world0.shape[0]):
                for y in range(world0.shape[1]):
                    if world0[x,y]==0:
                        return x,y
                    
        def floodfill(world,i,j):
            sx,sy=world.shape[0],world.shape[1]
            if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                return
            if(world[i,j]==-1):return
            world[i,j] = -1
            floodfill(world,i+1,j)
            floodfill(world,i,j+1)
            floodfill(world,i-1,j)
            floodfill(world,i,j-1)

        i,j = firstFree(world0)
        floodfill(world0,i,j)
        if np.any(world0==0):
            return False
        else:
            return True

    def getObstacleMap(self):
        return (self.world.state==-1).astype(int)
    
    def getGoals(self):
        result=[]
        for i in range(1,self.num_agents+1):
            result.append(self.world.getGoal(i))
        return result
    
    def getPositions(self):
        result=[]
        for i in range(1,self.num_agents+1):
            result.append(self.world.getPos(i))
        return result
    
    def _reset(self):
        self._create_world()
        # self._create_agent(num_agents = self.config["num_agents"])

    def _create_world(self):
        # a = np.zeros((5,5), dtype=int)
        # goal = np.zeros((5,5),dtype=int)
        # a[1,1]=-1
        # a[4,4]=1
        # a[1,3]=2
        # goal[4,1] = 1
        # goal[0,3] = 2

        # a[0,2]=-1
        # a[3,1]=1
        # a[4,1]=2
        # goal[4,4] = 1
        # goal[2,2] = 2

        # a[0,4],a[1,3],a[2,0],a[3,0]=-1,-1,-1,-1
        # a[0,0]=1
        # a[2,1]=2
        # goal[4,3] = 1
        # goal[0,1] = 2

        # a[1,3]=1
        # a[0,0]=1
        # a[1,4]=2
        # goal[4,3]=1
        # goal[3,2]=2
        # self.world = setWorld(self.config, world0=a, goals0=goal, blank_world=False)
        self.world = setWorld(self.config, world0=None, goals0=None, blank_world=False)
        self.agents = self.world.agents

    def _step(self, action_input, episode=0):
        #episode is an optional variable which will be used on the reward discounting
        self.fresh = False
        n_actions = 9 if self.DIAGONAL_MOVEMENT else 5
        
        # Check action input
        # assert len(action_input) == 2, 'Action input should be a tuple with the form (agent_id, action)'
        # assert action_input[1] in range(n_actions), 'Invalid action'
        # assert action_input[0] in range(1, self.num_agents+1)

        # Parse action input
        # agent_id = action_input[0]
        # action   = action_input[1]
        # agent_id = 1
        # action = action_input[0]

        # Lock mutex (race conditions start here)
        # self.mutex.acquire()

        #get start location of agent
        # agentStartLocation = self.world.getPos(agent_id)
        agentStartLocation = [agent.position for agent in self.agents]

        # Execute action & determine reward
        action_statuss = [self.world.act(action, agent.id_label) for action, agent in zip(action_input, self.agents)]
        
        valid_action= [action_status >=0 for action_status in action_statuss]
        #     2: action executed and left goal
        #     1: action executed and reached/stayed on goal
        #     0: action executed
        #    -1: out of bounds
        #    -2: collision with wall
        #    -3: collision with robot
        def reward(index, action, action_status):
            blocking=False
            if action==0:#staying still
                if action_status == 1:#stayed on goal
                    reward=GOAL_REWARD
                    x=self.get_blocking_reward(index+1)
                    reward+=x
                    if x<0:
                        blocking=True
                elif action_status == 0:#stayed off goal
                    reward=IDLE_COST
            else:#moving
                if (action_status == 1): # reached goal
                    reward = GOAL_REWARD
                elif (action_status == -3 or action_status==-2 or action_status==-1): # collision
                    reward = COLLISION_REWARD
                elif (action_status == 2): #left goal
                    reward=ACTION_COST
                else:
                    reward=ACTION_COST
            return reward, blocking
        
        r, b = zip(*[reward(index, action, action_status) for index, (action, action_status) in enumerate(zip(action_input, action_statuss))])
        self.individual_rewards = list(r)
        self.blocking = list(b)

        # self.individual_rewards[agent_id-1]=reward

        if JOINT:
            visible=[False for i in range(self.num_agents)]
            v=0
            #joint rewards based on proximity
            for agent in range(1,self.num_agents+1):
                #tally up the visible agents
                if agent==agent_id:
                    continue
                top_left=(self.world.getPos(agent_id)[0]-self.observation_size//2, \
                          self.world.getPos(agent_id)[1]-self.observation_size//2)
                pos=self.world.getPos(agent)
                if pos[0]>=top_left[0] and pos[0]<top_left[0]+self.observation_size\
                    and pos[1]>=top_left[1] and pos[1]<top_left[1]+self.observation_size:
                        #if the agent is within the bounds for observation
                        v+=1
                        visible[agent-1]=True
            if v>0:
                reward=self.individual_rewards[agent_id-1]/2
                #set the reward to the joint reward if we are 
                for i in range(self.num_agents):
                    if visible[i]:
                        reward+=self.individual_rewards[i]/(v*2)

        # Perform observation
        state = [self._observe(agent.id_label) for agent in self.agents]

        # Done?
        done = self.world.done()
        self.finished |= done

        # next valid actions
        nextActions = [self._listNextValidActions(agent.id_label, action, episode=episode)  for agent,action in zip(self.agents, action_input)]

        # on_goal estimation
        # on_goal = self.world.getPos(agent_id) == self.world.getGoal(agent_id)
        on_goal = [(agent.position == agent.goal) for agent in self.agents]
        
        # Unlock mutex
        # self.mutex.release()
        return state, self.individual_rewards, [done], nextActions, on_goal, self.blocking, valid_action
    
    def astar(self,world,start,goal,robots=[]):
        '''robots is a list of robots to add to the world'''
        for (i,j) in robots:
            world[i,j]=1
        try:
            path=cpp_mstar.find_path(world,[start],[goal],1,5)
        except NoSolutionError:
            path=None
        for (i,j) in robots:
            world[i,j]=0
        return path
    
    def get_blocking_reward(self,agent_id):
        '''calculates how many robots the agent is preventing from reaching goal
        and returns the necessary penalty'''
        #accumulate visible robots
        other_robots=[]
        other_locations=[]
        inflation=10
        top_left=(self.world.getPos(agent_id)[0]-self.observation_size//2,self.world.getPos(agent_id)[1]-self.observation_size//2)
        bottom_right=(top_left[0]+self.observation_size,top_left[1]+self.observation_size)        
        for agent in range(1,self.num_agents):
            if agent==agent_id: continue
            x,y=self.world.getPos(agent)
            if x<top_left[0] or x>=bottom_right[0] or y>=bottom_right[1] or y<top_left[1]:
                continue
            other_robots.append(agent)
            other_locations.append((x,y))
        num_blocking=0
        world=self.getObstacleMap()
        for agent in other_robots:
            other_locations.remove(tuple(self.world.getPos(agent)))
            #before removing
            path_before=self.astar(world,self.world.getPos(agent),self.world.getGoal(agent),
                                   robots=other_locations+[self.world.getPos(agent_id)])
            #after removing
            path_after=self.astar(world,self.world.getPos(agent),self.world.getGoal(agent),
                                   robots=other_locations)
            other_locations.append(self.world.getPos(agent))
            if (path_before is None and path_after is None):continue
            if (path_before is not None and path_after is None):continue
            if (path_before is None and path_after is not None)\
                or len(path_before)>len(path_after)+inflation:
                num_blocking+=1
        return num_blocking*BLOCKING_COST
    
    def _create_agent(self, num_agents):
        agents  = []
        for pos, goal in zip(self.world.agents, self.world.agent_goals):
            agent = Agent.create_random(
                    self.world,
                    position = np.array(pos),
                    heading=0, # float, 0 / 90 / 180 / 360
                    speed=2,
                    goal= np.array(goal),
                    # spacing=self.config["ego_spacing"],
                    ) 
            # agent = self.action_type.agent_class(
            #     self.world, agent.position, agent.heading, agent.speed
            #         )
            agents.append(agent)
        # self.controlled_vehicles.append(agent)
        self.world.agents = agents
        # self.agents_count = self.world.agents_count
        # self.observer.set_env(self.world)
    # def _reset(self, agent_id, world0=None,goals0=None):
        # self.finished = False
        # # self.mutex.acquire()
        # # Initialize data structures
        # self._setWorld(world0,goals0)
        # self.fresh = True

        # # self.mutex.release()
        # if self.viewer is not None:
        #     self.viewer = None

        # on_goal = self.world.getPos(agent_id) == self.world.getGoal(agent_id)
        # #we assume you don't start blocking anyone (the probability of this happening is insanely low)
        # return self._listNextValidActions(agent_id), on_goal, False


    def _listNextValidActions(self, agent_id, prev_action=0, episode=0):
        available_actions = [0] # staying still always allowed
        # opposite_actions = self.world.state.opposite_actions
        # Get current agent position
        agent_pos = self.world.getPos(agent_id)
        ax,ay     = agent_pos[0],agent_pos[1]
        n_moves   = 9 if self.DIAGONAL_MOVEMENT else 5

        for action in range(1,n_moves):
            direction = self.world.getDir(action)
            dx,dy     = direction[0],direction[1]
            if(ax+dx>=self.world.state.shape[0] or ax+dx<0 or ay+dy>=self.world.state.shape[1] or ay+dy<0):#out of bounds
                continue
            if(self.world.state[ax+dx,ay+dy]<0):#collide with static obstacle
                continue
            if(self.world.state[ax+dx,ay+dy]>0):#collide with robot
                continue
            # check for diagonal collisions
            if(self.DIAGONAL_MOVEMENT):
                if self.world.diagonalCollision(agent_id,(ax+dx,ay+dy)):
                    continue          
            #otherwise we are ok to carry out the action
            available_actions.append(action)

        if opposite_actions[prev_action] in available_actions:
            available_actions.remove(opposite_actions[prev_action])
                
        return available_actions
    
    def parse_path(self, path):
        '''needed function to take the path generated from M* and create the 
        observations and actions for the agent
        path: the exact path ouput by M*, assuming the correct number of agents
        returns: the list of rollouts for the "episode": 
                list of length num_agents with each sublist a list of tuples 
                (observation[0],observation[1],optimal_action,reward)'''
        result = [[] for i in range(self.num_agents)]
        step = 0
        for t in range(len(path[:-1])):
            move_queue = list(range(self.num_agents))
            # for agent in range(1, self.num_agents+1):
                # observations.append(self._observe(agent))
            observations = [self._observe(agent) for agent in range(1, self.num_agents+1)]
            step+=1
            poss,newPos = path[t], path[t+1]
            directions = [(newPos[i][0]-poss[i][0], newPos[i][1]-poss[i][1]) for i in range(self.num_agents)]
            a_s = [self.world.getAction(directions[i]) for i in range(self.num_agents)]
            state, reward, done, nextActions, on_goal, blocking, valid_action = self._step(a_s)
            
            if not all(valid_action):
                print(f"poss:{poss},newPos:{newPos},valid_action:{valid_action}")
                print("Invalid action, breaking")
                continue 
            for i in range(self.num_agents):
                result[i].append([observations[i], a_s[i]])
            # while len(move_queue) > 0:
            #     steps += 1
            #     i = move_queue.pop(0)
            #     o = observations[i]
            #     pos = path[t][i]
            #     # guaranteed to be in bounds by loop guard
            #     newPos = path[t+1][i]
            #     direction = (newPos[0]-pos[0], newPos[1]-pos[1])
            #     a = self.world.getAction(direction)
            #     state, reward, done, nextActions, on_goal, blocking, valid_action = self._step(
            #         (i+1, a))
            #     if steps > self.num_agents**2:
            #         # if we have a very confusing situation where lots of agents move
            #         # in a circle (difficult to parse and also (mostly) impossible to learn)
            #         return None
            #     if not valid_action:
            #         # the tie must be broken here
            #         move_queue.append(i)
            #         continue
            #     result[i].append([o['maps'], o['goal_vector'], a])
        return result

    
    # Returns an observation of an agent
    def _observe(self,agent_id):
        assert(agent_id>0)
        top_left=(self.world.getPos(agent_id)[0]-self.observation_size//2,self.world.getPos(agent_id)[1]-self.observation_size//2)
        bottom_right=(top_left[0]+self.observation_size,top_left[1]+self.observation_size)        
        obs_shape=(self.observation_size,self.observation_size)
        goal_map             = np.zeros(obs_shape)
        poss_map             = np.zeros(obs_shape)
        goals_map            = np.zeros(obs_shape)
        obs_map              = np.zeros(obs_shape)
        visible_agents=[]
        for i in range(top_left[0],top_left[0]+self.observation_size):
            for j in range(top_left[1],top_left[1]+self.observation_size):
                if i>=self.world.state.shape[0] or i<0 or j>=self.world.state.shape[1] or j<0:
                    #out of bounds, just treat as an obstacle
                    obs_map[i-top_left[0],j-top_left[1]]=1
                    continue
                if self.world.state[i,j]==-1:
                    #obstacles
                    obs_map[i-top_left[0],j-top_left[1]]=1
                if self.world.state[i,j]==agent_id:
                    #agent's position
                    poss_map[i-top_left[0],j-top_left[1]]=1
                if self.world.goals[i,j]==agent_id:
                    #agent's goal
                    goal_map[i-top_left[0],j-top_left[1]]=1
                if self.world.state[i,j]>0 and self.world.state[i,j]!=agent_id:
                    #other agents' positions
                    visible_agents.append(self.world.state[i,j])
                    poss_map[i-top_left[0],j-top_left[1]]=1

        for agent in visible_agents:
            x, y = self.world.getGoal(agent)
            min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.observation_size - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx=self.world.getGoal(agent_id)[0]-self.world.getPos(agent_id)[0]
        dy=self.world.getGoal(agent_id)[1]-self.world.getPos(agent_id)[1]
        mag=(dx**2+dy**2)**.5
        if mag!=0:
            dx=dx/mag
            dy=dy/mag
        # 当前agent的相对位置，目标的相对位置，所有agent目标的相对位置，和障碍物相对位置
            
        return {'maps':np.stack([poss_map, goal_map, goals_map, obs_map], axis=0), 'goal_vector': np.array([[dx,dy,mag]])}
        # return {"pos_map":poss_map,
        #         "goal_map":goal_map,
        #         "goals_map": goals_map,
        #         "obs_map": obs_map,
        #         "direction_vector": [dx,dy,mag]
        #         }
        # return ([poss_map,goal_map,goals_map,obs_map],[dx,dy,mag])


    