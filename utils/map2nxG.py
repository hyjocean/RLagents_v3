import networkx as nx
import numpy as np
# 假设 map_grid 是一个二维列表，其中 0 表示空格，1 表示障碍物
map_grid = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
]


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

def create_graph_from_map2(map_grid):
    # 创建一个5x10的网格图
    G = nx.grid_2d_graph(5, 10)

    # 假设障碍物的位置列表
    # obstacles = [(1, 1), (1, 6), (2, 3), (2, 5), (4, 1)]  # 示例障碍物位置
    obstacles = np.where(map_grid == 1)
    # 移除障碍物节点
    for obstacle in obstacles:
        if obstacle in G:
            G.remove_node(obstacle)
    return G



# 创建图
G = create_graph_from_map2(map_grid)
# 使用matplotlib可视化图形
# import matplotlib.pyplot as plt

# pos = {(x, y): (y, -x) for x, y in G.nodes()}
# nx.draw(G, pos=pos, with_labels=True, node_size=700, node_color="skyblue")
# plt.show()
spath = nx.shortest_path(G, (1,2), (4,3))
print(spath)