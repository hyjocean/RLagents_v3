import networkx as nx
import numpy as np
import copy
# 假设 map_grid 是一个二维列表，其中 0 表示空格，1 表示障碍物
from utils.lMRP import lmrp_find_path


def create_graph_from_map(map_grid):
    rows = len(map_grid)
    cols = len(map_grid[0]) if rows else 0

    G = nx.Graph()
    
    for i in range(rows):
        for j in range(cols):
            # 只有当格子不是障碍物时，才添加节点
            if map_grid[i][j] == 0:
                G.add_node((i, j))
                # 连接上下左右的节点
                if i > 0 and map_grid[i - 1][j] == 0:
                    G.add_edge((i, j), (i - 1, j))
                if i < rows - 1 and map_grid[i + 1][j] == 0:
                    G.add_edge((i, j), (i + 1, j))
                if j > 0 and map_grid[i][j - 1] == 0:
                    G.add_edge((i, j), (i, j - 1))
                if j < cols - 1 and map_grid[i][j + 1] == 0:
                    G.add_edge((i, j), (i, j + 1))
    return G

def create_graph_from_map2(world_state):
    # 创建一个5x10的网格图

    world_state[world_state>0] =0
    world_state[world_state<0] =1
    map_grid = np.array(world_state)
    G = nx.grid_2d_graph(*map_grid.shape)

    # 假设障碍物的位置列表
    # obstacles = [(1, 1), (1, 6), (2, 3), (2, 5), (4, 1)]  # 示例障碍物位置
    obstacles = np.transpose(np.where(map_grid == 1))
    # 移除障碍物节点
    for obstacle in obstacles:
        if tuple(obstacle) in G:
            G.remove_node(tuple(obstacle))
    return G


def nx_generate_path(world_state, agents_pos, agents_goal):
    no_path, blck_path = 1,1
    world_G = create_graph_from_map2(world_state)
    for agent_pos, agent_goal in zip(agents_pos, agents_goal):
        preprocess_G = copy.deepcopy(world_G)
    paths = [nx.shortest_paths.astar.astar_path(world_G, tuple(agent_pos), tuple(agent_goal)) for agent_pos, agent_goal in zip(agents_pos, agents_goal)]
    if None in paths:
        print("No path found for some agents")
        np_path = 0
        return no_path
    # 确定最长路径的长度
    max_length = max(len(p) for p in paths)
    padded_paths = []
    for i in range(max_length):
        # 如果路径长度小于最大长度，用最后一个元素进行填充
        path = [p[i] if i < len(p) else p[-1] for p in paths]
        if path[0] == path[1]:
            blck_path = 0
            return blck_path
        if i > 1 and path[0] == padded_paths[-1][1]:
            padded_paths.append([padded_paths[-1][0], path[1]])
        if i> 1 and path[1] == padded_paths[-1][0]:
            padded_paths.append([path[0], padded_paths[-1][1]])
        padded_paths.append(path)
    return padded_paths


def lmrp_generate_path(world_state, agents_pos, agents_goal,agents_name, path_dict):
    path = lmrp_find_path(world_state, agents_pos, agents_goal,agents_name, path_dict)
    # path_0 = list(map(lambda x: tuple(x.tolist()), agents_pos)) # 起始位置
    # path = [path_0] + path
    return path


if __name__ == "__main__":
    map_grid = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
    ]

    # 创建图
    G = create_graph_from_map2(nmap_grid)
    # 使用matplotlib可视化图形
    # import matplotlib.pyplot as plt

    # pos = {(x, y): (y, -x) for x, y in G.nodes()}
    # nx.draw(G, pos=pos, with_labels=True, node_size=700, node_color="skyblue")
    # plt.show()
    spath = nx.shortest_path(G, (1,2), (4,3))
    print(spath)