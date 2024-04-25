import numpy as np
import pandas as pd 
import yaml
import os
import subprocess
from pathlib import Path


def write_file(world_state, agents_pos, agents_goal, agents_name, path_dict):
    file_name = Path(path_dict['ref_path_file']).joinpath(f"{path_dict['path_id']}.yaml")
    map_size = world_state.shape
    obstacles = np.transpose(np.where(world_state == -1))
    with open(file_name, 'w') as file:
        file.write(f"agents:\n")
        for pos, goal, name in zip(agents_pos, agents_goal, agents_name):
            file.write(f"-   goal: {str(list(goal))}\n")
            file.write(f"    name: agent_{name}\n")
            file.write(f"    start: {str(list(pos))}\n")
            
        file.write(f"map:\n")
        file.write(f"    dimensions: {str(list(map_size))}\n")
        file.write(f"    obstacles: \n")
        if obstacles is not None:
            for obstacle in obstacles:
                file.write(f"    - {str(list(obstacle))}  \n")

        file.close()

def imrp_pathfinding(path_dict):
    in_f = Path(path_dict['ref_path_file']).joinpath(f"{path_dict['path_id']}.yaml")
    out_f = Path(path_dict['ref_path_file']).joinpath(f"{path_dict['path_id']}_out.yaml")
    # in_f = f"/home/bld/HK_RL/RLagents_v3/lMRP_paths/input_map/{input_file}.yaml"
    # out_f = f"/home/bld/HK_RL/RLagents_v3/lMRP_paths/output_res/{input_file}_out.yaml"
    command = [path_dict['cbs_file'],
            "-i", in_f,
            "-o", out_f]
    # command = ["/home/bld/HK_RL/RLagents_v3/libMultiRobotPlanning/build/cbs",
    #             "-i", in_f,
    #             "-o", out_f]
    
    res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # os.system(f"/home/bld/HK_RL/RLagents_v3/libMultiRobotPlanning/build/cbs -i  {in_f} -o {out_f}")
    return out_f

def get_res(res_file):
    # out_f = f"/home/bld/HK_RL/RLagents_v3/lMRP_paths/output_res/{inpu}_out.yaml"
    with open(res_file, 'r') as file:
        res = yaml.load(file, Loader=yaml.FullLoader)

    
    return res

def lmrp_find_path(world_state, agents_pos, agents_goal, agents_name, path_dict):
    write_file(world_state, agents_pos, agents_goal, agents_name, path_dict)
    out_f = imrp_pathfinding(path_dict)
    res = get_res(out_f)
    max_span = res['statistics']['makespan']
    path = [[] for _ in range(max_span+1)]
    for keys in res['schedule']:
        single_agent = res['schedule'][keys]
        for i in range(max_span+1):
            path[i].append((single_agent[i]['x'],single_agent[i]['y']) if i < len(single_agent) else (single_agent[-1]['x'],single_agent[-1]['y']))
            # assert single_agent[i]['t'] == i, f"path sequence index {i} not equal to imrp path t {single_agent[i]['t']}"
    return path

