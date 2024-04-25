import sys
import random
import numpy as np
import logging
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional
import copy
import math
import time
import warnings
# import imageio
import gymnasium as gym
from gymnasium import spaces
# from matplotlib.colors import *
from operator import sub, add
from ..agents.kinematics import Agent

# from highway_env.road.lane import LineType, StraightLane, AbstractLane, lane_from_config
# from highway_env.vehicle.objects import Landmark

# if TYPE_CHECKING:
#     from highway_env.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

# LaneIndex = Tuple[str, str, int]
# Route = List[LaneIndex]


# def make_gif(images, fname):
#     gif = imageio.mimwrite(fname, images, subrectangles=True)
#     print("wrote gif")
#     return gif


def opposite_actions(action, isDiagonal=False):
    if isDiagonal:
        checking_table = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2}
        raise NotImplemented
    else:
        checking_table = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2}
    return checking_table[action]


def action2dir(action):
    checking_table = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    return checking_table[action]


def dir2action(direction):
    checking_table = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}
    return checking_table[direction]


def tuple_plus(a, b):
    """ a + b """
    return tuple(map(add, a, b))


def tuple_minus(a, b):
    """ a - b """
    return tuple(map(sub, a, b))


def _heap(ls, max_length):
    while True:
        if len(ls) > max_length:
            ls.pop(0)
        else:
            return ls


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def getAstarDistanceMap(map: np.array, start: tuple, goal: tuple, isDiagonal: bool = False):
    """
    returns a numpy array of same dims as map with the distance to the goal from each coord
    :param map: a n by m np array, where -1 denotes obstacle
    :param start: start_position
    :param goal: goal_position
    :return: optimal distance map
    """

    def lowestF(fScore, openSet):
        # find entry in openSet with lowest fScore
        assert (len(openSet) > 0)
        minF = 2 ** 31 - 1
        minNode = None
        for (i, j) in openSet:
            if (i, j) not in fScore:
                continue
            if fScore[(i, j)] < minF:
                minF = fScore[(i, j)]
                minNode = (i, j)
        return minNode

    def getNeighbors(node):
        # return set of neighbors to the given node
        n_moves = 9 if isDiagonal else 5
        neighbors = set()
        for move in range(1, n_moves):  # we dont want to include 0 or it will include itself
            direction = action2dir(move)
            dx = direction[0]
            dy = direction[1]
            ax = node[0]
            ay = node[1]
            if (ax + dx >= map.shape[0] or ax + dx < 0 or ay + dy >= map.shape[
                    1] or ay + dy < 0):  # out of bounds
                continue
            if map[ax + dx, ay + dy] == -1:  # collide with static obstacle
                continue
            neighbors.add((ax + dx, ay + dy))
        return neighbors

    # NOTE THAT WE REVERSE THE DIRECTION OF SEARCH SO THAT THE GSCORE WILL BE DISTANCE TO GOAL
    start, goal = goal, start
    start, goal = tuple(start), tuple(goal)
    # The set of nodes already evaluated
    closedSet = set()

    # The set of currently discovered nodes that are not evaluated yet.
    # Initially, only the start node is known.
    openSet = set()
    openSet.add(start)

    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    cameFrom = dict()

    # For each node, the cost of getting from the start node to that node.
    gScore = dict()  # default value infinity

    # The cost of going from start to start is zero.
    gScore[start] = 0

    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    fScore = dict()  # default infinity

    # our heuristic is euclidean distance to goal
    def heuristic_cost_estimate(x, y): return math.hypot(
        x[0] - y[0], x[1] - y[1])

    # For the first node, that value is completely heuristic.
    fScore[start] = heuristic_cost_estimate(start, goal)

    while len(openSet) != 0:
        # current = the node in openSet having the lowest fScore value
        current = lowestF(fScore, openSet)

        openSet.remove(current)
        closedSet.add(current)
        for neighbor in getNeighbors(current):
            if neighbor in closedSet:
                continue  # Ignore the neighbor which is already evaluated.

            if neighbor not in openSet:  # Discover a new node
                openSet.add(neighbor)

            # The distance from start to a neighbor
            # in our case the distance between is always 1
            tentative_gScore = gScore[current] + 1
            if tentative_gScore >= gScore.get(neighbor, 2 ** 31 - 1):
                continue  # This is not a better path.

            # This path is the best until now. Record it!
            cameFrom[neighbor] = current
            gScore[neighbor] = tentative_gScore
            fScore[neighbor] = gScore[neighbor] + \
                heuristic_cost_estimate(neighbor, goal)

            # parse through the gScores
    Astar_map = map.copy()
    for (i, j) in gScore:
        Astar_map[i, j] = gScore[i, j]
    return Astar_map


def setWorld(config, world0=None, goals0=None,blank_world=False):
    #blank_world is a flag indicating that the world given has no agent or goal positions 
    PROB = config['PROB']
    SIZE = config['SIZE']
    DIAGONAL_MOVEMENT = config['DIAGONAL_MOVEMENT']
    num_agents = config['num_agents']
    def getConnectedRegion(world,regions_dict,x,y):
        sys.setrecursionlimit(1000000)
        '''returns a list of tuples of connected squares to the given tile
        this is memoized with a dict'''
        if (x,y) in regions_dict:
            return regions_dict[(x,y)]
        visited=set()
        sx,sy=world.shape[0],world.shape[1]
        work_list=[(x,y)]
        while len(work_list)>0:
            (i,j)=work_list.pop()
            if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                continue
            if(world[i,j]==-1):
                continue#crashes
            if world[i,j]>0:
                regions_dict[(i,j)]=visited
            if (i,j) in visited:continue
            visited.add((i,j))
            work_list.append((i+1,j))
            work_list.append((i,j+1))
            work_list.append((i-1,j))
            work_list.append((i,j-1))
        regions_dict[(x,y)]=visited
        return visited
    #defines the State object, which includes initializing goals and agents
    #sets the world to world0 and goals, or if they are None randomizes world
    if not (world0 is None):
        if goals0 is None and not blank_world:
            raise Exception("you gave a world with no goals!")
        if blank_world:
            #RANDOMIZE THE POSITIONS OF AGENTS
            agent_counter = 1
            agent_locations=[]
            while agent_counter<=num_agents:
                x,y       = np.random.randint(0,world0.shape[0]),np.random.randint(0,world0.shape[1])
                if(world0[x,y] == 0):
                    world0[x,y]=agent_counter
                    agent_locations.append((x,y))
                    agent_counter += 1   
            #RANDOMIZE THE GOALS OF AGENTS
            goals0 = np.zeros(world0.shape).astype(int)
            goal_counter = 1
            agent_regions=dict()  
            while goal_counter<=num_agents:
                agent_pos=agent_locations[goal_counter-1]
                valid_tiles=getConnectedRegion(world0,agent_regions,agent_pos[0],agent_pos[1])#crashes
                x,y  = random.choice(list(valid_tiles))
                if(goals0[x,y]==0 and world0[x,y]!=-1):
                    goals0[x,y]    = goal_counter
                    goal_counter += 1
            initial_world = world0.copy()
            initial_goals = goals0.copy()
            world = State(initial_world,initial_goals,DIAGONAL_MOVEMENT,num_agents)
            return
        initial_world = world0
        initial_goals = goals0
        world = State(world0,goals0,DIAGONAL_MOVEMENT,num_agents)
        return world
    #otherwise we have to randomize the world
    #RANDOMIZE THE STATIC OBSTACLES
    prob=np.random.triangular(PROB[0],.33*PROB[0]+.66*PROB[1],PROB[1])
    size=np.random.choice([SIZE[0],SIZE[0]*.5+SIZE[1]*.5,SIZE[1]],p=[.5,.25,.25])
    world     = -(np.random.rand(int(size),int(size))<prob).astype(int)
    #RANDOMIZE THE POSITIONS OF AGENTS
    agent_counter = 1
    agent_locations=[]
    while agent_counter<=num_agents:
        x,y       = np.random.randint(0,world.shape[0]),np.random.randint(0,world.shape[1])
        if(world[x,y] == 0):
            world[x,y]=agent_counter
            agent_locations.append((x,y))
            agent_counter += 1        
    
    #RANDOMIZE THE GOALS OF AGENTS
    goals = np.zeros(world.shape).astype(int)
    goal_counter = 1
    agent_regions=dict()     
    while goal_counter<=num_agents:
        agent_pos=agent_locations[goal_counter-1]
        valid_tiles=getConnectedRegion(world,agent_regions,agent_pos[0],agent_pos[1])
        x,y  = random.choice(list(valid_tiles))
        if(goals[x,y]==0 and world[x,y]!=-1):
            goals[x,y]    = goal_counter
            goal_counter += 1
    initial_world = world
    initial_goals = goals
    world = State(world, goals, DIAGONAL_MOVEMENT,num_agents=num_agents)
    return world

class State(object):
    '''
    State.
    Implemented as 2 2d numpy arrays.
    first one "state":
        static obstacle: -1
        empty: 0
        agent = positive integer (agent_id)
    second one "goals":
        agent goal = positive int(agent_id)
    '''
    ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD,FINISH_REWARD,BLOCKING_COST = -0.3, -.5, 0.0, -2.,20.,-1.
    opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
    JOINT = False # True for joint estimation of rewards for closeby agents
    dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
    actionDict={v:k for k,v in dirDict.items()}

    def __init__(self, world0, goals, diagonal, num_agents=1):
        assert(len(world0.shape) == 2 and world0.shape==goals.shape)
        self.state                    = world0.copy()
        self.goals                    = goals.copy()
        self.num_agents               = num_agents
        self.agents, self.agents_past, self.agent_goals = self.scanForAgents()
        self.diagonal=diagonal
        assert(len(self.agents) == num_agents)

    def scanForAgents(self):
        agents = [Agent(world=self.state, id_label=i+1) for i in range(self.num_agents)]
        agents_last = [(-1,-1) for i in range(self.num_agents)]        
        agent_goals = [(-1,-1) for i in range(self.num_agents)]        
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if(self.state[i,j]>0):
                    agents[self.state[i,j]-1].position = np.asarray((i,j))
                    agents[self.state[i,j]-1].start_pos = np.asarray((i,j))
                    agents[self.state[i,j]-1].last = np.asarray((i,j))
                    # agents[self.state[i,j]-1] = (i,j)
                    agents_last[self.state[i,j]-1] = (i,j)
                    
                if(self.goals[i,j]>0):
                    agents[self.goals[i,j]-1].goal = np.asarray((i,j))
                    agent_goals[self.goals[i,j]-1] = (i,j)
        # assert [(agent.position and agent.goal).any for agent in agents]
        assert ((-1,-1) not in agents_last and (-1,-1) not in agent_goals)
        # assert agent_goals
        return agents, agents_last, agent_goals

    def getPos(self, agent_id):

        if isinstance(self.agents[0], Agent):
            return self.agents[agent_id-1].position
        else:
            return self.agents[agent_id-1]

    def getPastPos(self, agent_id):
        return self.agents_past[agent_id-1]

    def getGoal(self, agent_id):
        return self.agent_goals[agent_id-1]
    
    def diagonalCollision(self, agent_id, newPos):
        '''diagonalCollision(id,(x,y)) returns true if agent with id "id" collided diagonally with 
        any other agent in the state after moving to coordinates (x,y)
        agent_id: id of the desired agent to check for
        newPos: coord the agent is trying to move to (and checking for collisions)
        '''
#        def eq(f1,f2):return abs(f1-f2)<0.001
        def collide(a1,a2,b1,b2):
            '''
            a1,a2 are coords for agent 1, b1,b2 coords for agent 2, returns true if these collide diagonally
            '''
            return np.isclose( (a1[0]+a2[0]) /2. , (b1[0]+b2[0])/2. ) and np.isclose( (a1[1]+a2[1])/2. , (b1[1]+b2[1])/2. )
        assert(len(newPos) == 2);
        #up until now we haven't moved the agent, so getPos returns the "old" location
        lastPos = self.getPos(agent_id)
        for agent in range(1,self.num_agents+1):
            if agent == agent_id: continue
            aPast = self.getPastPos(agent)
            aPres = self.getPos(agent)
            if collide(aPast,aPres,lastPos,newPos): return True
        return False

    #try to move agent and return the status
    def moveAgent(self, direction, agent_id):
        
        ax=self.agents[agent_id-1].position[0] if isinstance(self.agents[0], Agent) else self.agents[agent_id-1][0]
        ay=self.agents[agent_id-1].position[1]  if isinstance(self.agents[0], Agent) else self.agents[agent_id-1][1]

        # Not moving is always allowed
        if(direction==(0,0)):
            self.agents_past[agent_id-1]=self.agents[agent_id-1]
            return 1 if self.goals[ax,ay]==agent_id else 0

        # Otherwise, let's look at the validity of the move
        dx,dy = direction[0], direction[1]
        if(ax+dx>=self.state.shape[0] or ax+dx<0 or ay+dy>=self.state.shape[1] or ay+dy<0):#out of bounds
            return -1
        if(self.state[ax+dx,ay+dy]<0):#collide with static obstacle
            return -2
        if(self.state[ax+dx,ay+dy]>0):#collide with robot
            return -3
        # check for diagonal collisions
        if(self.diagonal):
            if self.diagonalCollision(agent_id,(ax+dx,ay+dy)):
                return -3
        # No collision: we can carry out the action
        self.state[ax,ay] = 0
        self.state[ax+dx,ay+dy] = agent_id
        self.agents_past[agent_id-1]=self.agents[agent_id-1].position
        self.agents[agent_id-1].position = np.array((ax+dx,ay+dy))
        if self.goals[ax+dx,ay+dy]==agent_id:
            return 1
        elif self.goals[ax+dx,ay+dy]!=agent_id and self.goals[ax,ay]==agent_id:
            return 2
        else:
            return 0

    # try to execture action and return whether action was executed or not and why
    #returns:
    #     2: action executed and left goal
    #     1: action executed and reached goal (or stayed on)
    #     0: action executed
    #    -1: out of bounds
    #    -2: collision with wall
    #    -3: collision with robot
    def act(self, action, agent_id):
        # 0     1  2  3  4 
        # still N  E  S  W
        direction = self.getDir(action)
        moved = self.moveAgent(direction,agent_id)
        return moved

    def getDir(self,action):
        return self.dirDict[int(action)]
    
    def getAction(self,direction):  
        return self.actionDict[direction]

    # Compare with a plan to determine job completion
    def done(self):
        numComplete = 0
        for i in range(1,len(self.agents)+1):

            agent_pos = self.agents[i-1].position if isinstance(self.agents[0], Agent) else self.agents[i-1]
            if self.goals[agent_pos[0],agent_pos[1]] == i:
                numComplete += 1
        return numComplete==len(self.agents) #, numComplete/float(len(self.agents))


class World():
    """
    Include: basic world generation rules, blank map generation and collision checking.
    reset_world:
    Do not add action pruning, reward structure or any other routine for training in this class. Pls add in upper class MAPFEnv
    """

    def __init__(self, map_generator, agents_count, isDiagonal=False):
        self.agents_count = agents_count
        self.manual_world = False
        self.manual_goal = False
        self.goal_generate_distance = 2

        self.map_generator = map_generator
        self.isDiagonal = isDiagonal

        self.agents_init_pos, self.goals_init_pos = None, None
        self.reset_world()
        self.init_agents_and_goals()

    def reset_world(self):
        """
        generate/re-generate a world map, and compute its corridor map
        """

        def scan_for_agents(state_map):
            agents = {}
            for i in range(state_map.shape[0]):
                for j in range(state_map.shape[1]):
                    if state_map[i, j] > 0:
                        agentID = state_map[i, j]
                        agents.update({agentID: (i, j)})
            return agents

        self.state, self.goals_map = self.map_generator()
        # detect manual world
        if (self.state > 0).any():
            self.manual_world = True
            self.agents_init_pos = scan_for_agents(self.state)
            if self.agents_count is not None and self.agents_count != len(self.agents_init_pos.keys()):
                warnings.warn("num_agent does not match the actual agent number in manual map! "
                              "num_agent has been set to be consistent with manual map.")
            self.agents_count = len(self.agents_init_pos.keys())
            self.agents = {i: copy.deepcopy(Agent())
                           for i in range(1, self.agents_count + 1)}
        else:
            assert self.agents_count is not None
            self.agents = {i: copy.deepcopy(Agent())
                           for i in range(1, self.agents_count + 1)}
        # detect manual goals_map
        if self.goals_map is not None:
            self.manual_goal = True
            self.goals_init_pos = scan_for_agents(
                self.goals_map) if self.manual_goal else None

        else:
            self.goals_map = np.zeros(
                [self.state.shape[0], self.state.shape[1]])

        self.corridor_map = {}
        self.restrict_init_corridor = True
        self.visited = []
        self.corridors = {}
        self.get_corridors()

    def reset_agent(self):
        """
        remove all the agents (with their travel history) and goals in the env, rebase the env into a blank one
        """
        self.agents = {i: copy.deepcopy(Agent())
                       for i in range(1, self.agents_count + 1)}
        self.state[self.state > 0] = 0  # remove agents in the map

    def get_corridors(self):
        """
        in corridor_map , output = list:
            list[0] : if In corridor, corridor id , else -1 
            list[1] : If Inside Corridor = 1
                      If Corridor Endpoint = 2
                      If Free Cell Outside Corridor = 0   
                      If Obstacle = -1 
        """
        corridor_count = 1
        # Initialize corridor map
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if self.state[i, j] >= 0:
                    self.corridor_map[(i, j)] = [-1, 0]
                else:
                    self.corridor_map[(i, j)] = [-1, -1]
        # Compute All Corridors and End-points, store them in self.corridors , update corridor_map
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                positions = self.blank_env_valid_neighbor(i, j)
                if (positions.count(None)) == 2 and (i, j) not in self.visited:
                    allowed = self.check_for_singular_state(positions)
                    if not allowed:
                        continue
                    self.corridors[corridor_count] = {}
                    self.corridors[corridor_count]['Positions'] = [(i, j)]
                    self.corridor_map[(i, j)] = [corridor_count, 1]
                    self.corridors[corridor_count]['EndPoints'] = []
                    self.visited.append((i, j))
                    for num in range(4):
                        if positions[num] is not None:
                            self.visit(
                                positions[num][0], positions[num][1], corridor_count)
                    corridor_count += 1
        # Get Delta X , Delta Y for the computed corridors ( Delta= Displacement to corridor exit)
        for k in range(1, corridor_count):
            if k in self.corridors:
                if len(self.corridors[k]['EndPoints']) == 2:
                    self.corridors[k]['DeltaX'] = {}
                    self.corridors[k]['DeltaY'] = {}
                    pos_a = self.corridors[k]['EndPoints'][0]
                    pos_b = self.corridors[k]['EndPoints'][1]
                    # / (max(1, abs(pos_a[0] - pos_b[0])))
                    self.corridors[k]['DeltaX'][pos_a] = (pos_a[0] - pos_b[0])
                    self.corridors[k]['DeltaX'][pos_b] = - \
                        1 * self.corridors[k]['DeltaX'][pos_a]
                    # / (max(1, abs(pos_a[1] - pos_b[1])))
                    self.corridors[k]['DeltaY'][pos_a] = (pos_a[1] - pos_b[1])
                    self.corridors[k]['DeltaY'][pos_b] = - \
                        1 * self.corridors[k]['DeltaY'][pos_a]
            else:
                print('Weird2')

                # Rearrange the computed corridor list such that it becomes easier to iterate over the structure
        # Basically, sort the self.corridors['Positions'] list in a way that the first element of the list is
        # adjacent to Endpoint[0] and the last element of the list is adjacent to EndPoint[1]
        # If there is only 1 endpoint, the sorting doesn't matter since blocking is easy to compute
        for t in range(1, corridor_count):
            positions = self.blank_env_valid_neighbor(self.corridors[t]['EndPoints'][0][0],
                                                      self.corridors[t]['EndPoints'][0][1])
            for position in positions:
                if position is not None and self.corridor_map[position][0] == t:
                    break
            index = self.corridors[t]['Positions'].index(position)

            if index == 0:
                pass
            if index != len(self.corridors[t]['Positions']) - 1:
                temp_list = self.corridors[t]['Positions'][0:index + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]['Positions'][index + 1:]
                self.corridors[t]['Positions'] = []
                self.corridors[t]['Positions'].extend(temp_list)
                self.corridors[t]['Positions'].extend(temp_end)

            elif index == len(self.corridors[t]['Positions']) - 1 and len(self.corridors[t]['EndPoints']) == 2:
                positions2 = self.blank_env_valid_neighbor(self.corridors[t]['EndPoints'][1][0],
                                                           self.corridors[t]['EndPoints'][1][1])
                for position2 in positions2:
                    if position2 is not None and self.corridor_map[position2][0] == t:
                        break
                index2 = self.corridors[t]['Positions'].index(position2)
                temp_list = self.corridors[t]['Positions'][0:index2 + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]['Positions'][index2 + 1:]
                self.corridors[t]['Positions'] = []
                self.corridors[t]['Positions'].extend(temp_list)
                self.corridors[t]['Positions'].extend(temp_end)
                self.corridors[t]['Positions'].reverse()
            else:
                if len(self.corridors[t]['EndPoints']) == 2:
                    print("Weird3")

            self.corridors[t]['StoppingPoints'] = []
            if len(self.corridors[t]['EndPoints']) == 2:
                position_first = self.corridors[t]['Positions'][0]
                position_last = self.corridors[t]['Positions'][-1]
                self.corridors[t]['StoppingPoints'].append(
                    [position_first[0], position_first[1]])
                self.corridors[t]['StoppingPoints'].append(
                    [position_last[0], position_last[1]])
            else:
                position_first = self.corridors[t]['Positions'][0]
                self.corridors[t]['StoppingPoints'].append(
                    [position[0], position[1]])
                self.corridors[t]['StoppingPoints'].append(None)
        return

    def check_for_singular_state(self, positions):
        counter = 0
        for num in range(4):
            if positions[num] is not None:
                new_positions = self.blank_env_valid_neighbor(
                    positions[num][0], positions[num][1])
                if new_positions.count(None) in [2, 3]:
                    counter += 1
        return counter > 0

    def visit(self, i, j, corridor_id):
        positions = self.blank_env_valid_neighbor(i, j)
        if positions.count(None) in [0, 1]:
            self.corridors[corridor_id]['EndPoints'].append((i, j))
            self.corridor_map[(i, j)] = [corridor_id, 2]
            return
        elif positions.count(None) in [2, 3]:
            self.visited.append((i, j))
            self.corridors[corridor_id]['Positions'].append((i, j))
            self.corridor_map[(i, j)] = [corridor_id, 1]
            for num in range(4):
                if positions[num] is not None and positions[num] not in self.visited:
                    self.visit(positions[num][0],
                               positions[num][1], corridor_id)
        else:
            print('Weird')

    def blank_env_valid_neighbor(self, i, j):
        possible_positions = [None, None, None, None]
        move = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        if self.state[i, j] == -1:
            return possible_positions
        else:
            for num in range(4):
                x = i + move[num][0]
                y = j + move[num][1]
                if 0 <= x < self.state.shape[0] and 0 <= y < self.state.shape[1]:
                    if self.state[x, y] != -1:
                        possible_positions[num] = (x, y)
                        continue
        return possible_positions

    def getPos(self, agent_id):
        return tuple(self.agents[agent_id].position)

    def getDone(self, agentID):
        # get the number of goals that an agent has finished
        return self.agents[agentID].dones

    def get_history(self, agent_id, path_id=None):
        """
        :param: path_id: if None, get the last step
        :return: past_pos: (x,y), past_direction: int
        """

        if path_id is None:
            path_id = self.agents[agent_id].path_count - \
                1 if self.agents[agent_id].path_count > 0 else 0
        try:
            return self.agents[agent_id].position_history[path_id], self.agents[agent_id].direction_history[path_id]
        except IndexError:
            print("you are giving an invalid path_id")

    def getGoal(self, agent_id):
        return tuple(self.agents[agent_id].goal_pos)

    def init_agents_and_goals(self):
        """
        place all agents and goals in the blank env. If turning on corridor population restriction, only 1 agent is
        allowed to be born in each corridor.
        """

        def corridor_restricted_init_poss(state_map, corridor_map, goal_map, id_list=None):
            """
            generate agent init positions when corridor init population is restricted
            return a dict of positions {agentID:(x,y), ...}
            """
            if id_list is None:
                id_list = list(range(1, self.agents_count + 1))

            free_space1 = list(np.argwhere(state_map == 0))
            free_space1 = [tuple(pos) for pos in free_space1]
            corridors_visited = []
            manual_positions = {}
            break_completely = False
            for idx in id_list:
                if break_completely:
                    return None
                pos_set = False
                agentID = idx
                while not pos_set:
                    try:
                        assert (len(free_space1) > 1)
                        random_pos = np.random.choice(len(free_space1))
                    except AssertionError or ValueError:
                        print('wrong agent')
                        self.reset_world()
                        self.init_agents_and_goals()
                        break_completely = True
                        if idx == id_list[-1]:
                            return None
                        break
                    position = free_space1[random_pos]
                    cell_info = corridor_map[position[0], position[1]][1]
                    if cell_info in [0, 2]:
                        if goal_map[position[0], position[1]] != agentID:
                            manual_positions.update(
                                {idx: (position[0], position[1])})
                            free_space1.remove(position)
                            pos_set = True
                    elif cell_info == 1:
                        corridor_id = corridor_map[position[0], position[1]][0]
                        if corridor_id not in corridors_visited:
                            if goal_map[position[0], position[1]] != agentID:
                                manual_positions.update(
                                    {idx: (position[0], position[1])})
                                corridors_visited.append(corridor_id)
                                free_space1.remove(position)
                                pos_set = True
                        else:
                            free_space1.remove(position)
                    else:
                        print("Very Weird")
                        # print('Manual Positions' ,manual_positions)
            return manual_positions

        # no corridor population restriction
        if not self.restrict_init_corridor or (self.restrict_init_corridor and self.manual_world):
            self.put_goals(list(range(1, self.agents_count + 1)),
                           self.goals_init_pos)
            self._put_agents(
                list(range(1, self.agents_count + 1)), self.agents_init_pos)
        # has corridor population restriction
        else:
            check = self.put_goals(
                list(range(1, self.agents_count + 1)), self.goals_init_pos)
            if check is not None:
                manual_positions = corridor_restricted_init_poss(
                    self.state, self.corridor_map, self.goals_map)
                if manual_positions is not None:
                    self._put_agents(
                        list(range(1, self.agents_count + 1)), manual_positions)

    def _put_agents(self, id_list, manual_pos=None):
        """
        put some agents in the blank env, saved history data in self.agents and self.state
        get distance map for the agents
        :param id_list: a list of agent_id
                manual_pos: a dict of manual positions {agentID: (x,y),...}
        """
        if manual_pos is None:
            # randomly init agents everywhere
            free_space = np.argwhere(np.logical_or(
                self.state == 0, self.goals_map == 0) == 1)
            new_idx = np.random.choice(
                len(free_space), size=len(id_list), replace=False)
            init_poss = [free_space[idx] for idx in new_idx]
        else:
            assert len(manual_pos.keys()) == len(id_list)
            init_poss = [manual_pos[agentID] for agentID in id_list]
        assert len(init_poss) == len(id_list)
        for idx, agentID in enumerate(id_list):
            self.agents[agentID].ID = agentID
            self.agents_init_pos = {}
            if self.state[init_poss[idx][0], init_poss[idx][1]] in [0, agentID] \
                    and self.goals_map[init_poss[idx][0], init_poss[idx][1]] != agentID:
                self.state[init_poss[idx][0], init_poss[idx][1]] = agentID
                self.agents_init_pos.update(
                    {agentID: (init_poss[idx][0], init_poss[idx][1])})
            else:
                print(self.state)
                print(init_poss)
                raise ValueError('invalid manual_pos for agent' +
                                 str(agentID) + ' at: ' + str(init_poss[idx]))
            self.agents[agentID].move(init_poss[idx])
            self.agents[agentID].distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].position,
                                                                   self.agents[agentID].goal_pos)

    def put_goals(self, id_list, manual_pos=None):
        """
        put a goal of single agent in the env, if the goal already exists, remove that goal and put a new one
        :param manual_pos: a dict of manual_pos {agentID: (x, y)}
        :param id_list: a list of agentID
        :return: an Agent object
        """

        def random_goal_pos(previous_goals=None, distance=None):
            next_goal_buffer = {agentID: self.agents[agentID].next_goal for agentID in range(
                1, self.agents_count + 1)}
            curr_goal_buffer = {agentID: self.agents[agentID].goal_pos for agentID in range(
                1, self.agents_count + 1)}
            if previous_goals is None:
                previous_goals = {agentID: None for agentID in id_list}
            if distance is None:
                distance = self.goal_generate_distance
            free_for_all = np.logical_and(self.state == 0, self.goals_map == 0)
            # print(previous_goals)
            if not all(previous_goals.values()):  # they are new born agents
                free_space = np.argwhere(free_for_all == 1)
                init_idx = np.random.choice(
                    len(free_space), size=len(id_list), replace=False)
                new_goals = {agentID: tuple(
                    free_space[init_idx[agentID - 1]]) for agentID in id_list}
                return new_goals
            else:
                new_goals = {}
                for agentID in id_list:
                    free_on_agents = np.logical_and(
                        self.state > 0, self.state != agentID)
                    free_spaces_for_previous_goal = np.logical_or(
                        free_on_agents, free_for_all)
                    # free_spaces_for_previous_goal = np.logical_and(free_spaces_for_previous_goal, self.goals_map==0)
                    if distance > 0:
                        previous_x, previous_y = previous_goals[agentID]
                        x_lower_bound = (
                            previous_x - distance) if (previous_x - distance) > 0 else 0
                        x_upper_bound = previous_x + distance + 1
                        y_lower_bound = (
                            previous_y - distance) if (previous_x - distance) > 0 else 0
                        y_upper_bound = previous_y + distance + 1
                        free_spaces_for_previous_goal[x_lower_bound:x_upper_bound,
                                                      y_lower_bound:y_upper_bound] = False
                    free_spaces_for_previous_goal = list(
                        np.argwhere(free_spaces_for_previous_goal == 1))
                    free_spaces_for_previous_goal = [
                        pos.tolist() for pos in free_spaces_for_previous_goal]

                    try:
                        unique = False
                        counter = 0
                        while unique == False and counter < 500:
                            init_idx = np.random.choice(
                                len(free_spaces_for_previous_goal))
                            init_pos = free_spaces_for_previous_goal[init_idx]
                            unique = True
                            if tuple(init_pos) in next_goal_buffer.values() or tuple(
                                    init_pos) in curr_goal_buffer.values() or tuple(init_pos) in new_goals.values():
                                unique = False
                            if previous_goals is not None:
                                if tuple(init_pos) in previous_goals.values():
                                    unique = False
                            counter += 1
                        if counter >= 500:
                            print('Hard to find Non Conflicting Goal')
                        new_goals.update({agentID: tuple(init_pos)})
                    except ValueError:
                        print('wrong goal')
                        self.reset_world()
                        print(self.agents[1].position)
                        self.init_agents_and_goals()
                        return None
                return new_goals

        previous_goals = {
            agentID: self.agents[agentID].goal_pos for agentID in id_list}
        if manual_pos is None:
            new_goals = random_goal_pos(
                previous_goals, distance=self.goal_generate_distance)
        else:
            new_goals = manual_pos
        if new_goals is not None:  # recursive breaker
            refresh_distance_map = False
            for agentID in id_list:
                if self.state[new_goals[agentID][0], new_goals[agentID][1]] >= 0:
                    if self.agents[agentID].next_goal is None:  # no next_goal to use
                        # set goals_map
                        self.goals_map[new_goals[agentID][0],
                                       new_goals[agentID][1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = (
                            new_goals[agentID][0], new_goals[agentID][1])
                        # set agent.next_goal
                        new_next_goals = random_goal_pos(
                            new_goals, distance=self.goal_generate_distance)
                        if new_next_goals is None:
                            return None
                        self.agents[agentID].next_goal = (
                            new_next_goals[agentID][0], new_next_goals[agentID][1])
                        # remove previous goal
                        if previous_goals[agentID] is not None:
                            self.goals_map[previous_goals[agentID]
                                           [0], previous_goals[agentID][1]] = 0
                    else:  # use next_goal as new goal
                        # set goals_map
                        self.goals_map[self.agents[agentID].next_goal[0],
                                       self.agents[agentID].next_goal[1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = self.agents[agentID].next_goal
                        # set agent.next_goal
                        self.agents[agentID].next_goal = (
                            new_goals[agentID][0], new_goals[agentID][1])  # store new goal into next_goal
                        # remove previous goal
                        if previous_goals[agentID] is not None:
                            self.goals_map[previous_goals[agentID]
                                           [0], previous_goals[agentID][1]] = 0
                else:
                    print(self.state)
                    print(self.goals_map)
                    raise ValueError('invalid manual_pos for goal' +
                                     str(agentID) + ' at: ', str(new_goals[agentID]))
                if previous_goals[agentID] is not None:  # it has a goal!
                    if previous_goals[agentID] != self.agents[agentID].position:
                        print(self.state)
                        print(self.goals_map)
                        print(previous_goals)
                        raise RuntimeError(
                            "agent hasn't finished its goal but asking for a new goal!")

                    refresh_distance_map = True

                # compute distance map
                self.agents[agentID].next_distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].goal_pos,
                                                                            self.agents[agentID].next_goal)
                if refresh_distance_map:
                    self.agents[agentID].distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].position,
                                                                           self.agents[agentID].goal_pos)
            return 1
        else:
            return None

    def CheckCollideStatus(self, movement_dict):
        """
        WARNING: ONLY NON-DIAGONAL IS IMPLEMENTED
        return collision status and predicted next positions, do not move agent directly
        :return:
         1: action executed, and agents standing on its goal.
         0: action executed
        -1: collision with env (obstacles, out of bound)
        -2: collision with robot, swap
        -3: collision with robot, cell-wise
        """

        if self.isDiagonal is True:
            raise NotImplemented
        Assumed_newPos_dict = {}
        newPos_dict = {}
        status_dict = {agentID: None for agentID in range(
            1, self.agents_count + 1)}
        not_checked_list = list(range(1, self.agents_count + 1))

        # detect env collision
        for agentID in range(1, self.agents_count + 1):
            direction_vector = action2dir(movement_dict[agentID])
            newPos = tuple_plus(self.getPos(agentID), direction_vector)
            Assumed_newPos_dict.update({agentID: newPos})
            if newPos[0] < 0 or newPos[0] > self.state.shape[0] or newPos[1] < 0 \
                    or newPos[1] > self.state.shape[1] or self.state[newPos] == -1:
                status_dict[agentID] = -1
                newPos_dict.update({agentID: self.getPos(agentID)})
                Assumed_newPos_dict[agentID] = self.getPos(agentID)
                not_checked_list.remove(agentID)
                # collide, stay at the same place

        # detect swap collision

        for agentID in copy.deepcopy(not_checked_list):
            collided_ID = self.state[Assumed_newPos_dict[agentID]]
            if collided_ID != 0:  # some one is standing on the assumed pos
                # he wants to swap
                if Assumed_newPos_dict[collided_ID] == self.getPos(agentID):
                    if status_dict[agentID] is None:
                        status_dict[agentID] = -2
                        newPos_dict.update(
                            {agentID: self.getPos(agentID)})  # stand still
                        Assumed_newPos_dict[agentID] = self.getPos(agentID)
                        not_checked_list.remove(agentID)
                    if status_dict[collided_ID] is None:
                        status_dict[collided_ID] = -2
                        newPos_dict.update(
                            {collided_ID: self.getPos(collided_ID)})  # stand still
                        Assumed_newPos_dict[collided_ID] = self.getPos(
                            collided_ID)
                        not_checked_list.remove(collided_ID)

        # detect cell-wise collision
        for agentID in copy.deepcopy(not_checked_list):
            other_agents_dict = copy.deepcopy(Assumed_newPos_dict)
            other_agents_dict.pop(agentID)
            if Assumed_newPos_dict[agentID] in newPos_dict.values():
                status_dict[agentID] = -3
                newPos_dict.update(
                    {agentID: self.getPos(agentID)})  # stand still
                Assumed_newPos_dict[agentID] = self.getPos(agentID)
                not_checked_list.remove(agentID)
            elif Assumed_newPos_dict[agentID] in other_agents_dict.values():
                other_coming_agents = get_key(
                    Assumed_newPos_dict, Assumed_newPos_dict[agentID])
                other_coming_agents.remove(agentID)
                # if the agentID is the biggest among all other coming agents,
                # allow it to move. Else, let it stand still
                if agentID < min(other_coming_agents):
                    status_dict[agentID] = 1 if Assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos else 0
                    newPos_dict.update({agentID: Assumed_newPos_dict[agentID]})
                    not_checked_list.remove(agentID)
                else:
                    status_dict[agentID] = -3
                    newPos_dict.update(
                        {agentID: self.getPos(agentID)})  # stand still
                    Assumed_newPos_dict[agentID] = self.getPos(agentID)
                    not_checked_list.remove(agentID)

        # the rest are valid actions
        for agentID in copy.deepcopy(not_checked_list):
            status_dict[agentID] = 1 if Assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos else 0
            newPos_dict.update({agentID: Assumed_newPos_dict[agentID]})
            not_checked_list.remove(agentID)
        assert not not_checked_list

        return status_dict, newPos_dict
