import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import logging

def seed_set(SEED):
    if not SEED:
        SEED = np.random.randint(0, 10000)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.use_deterministic_algorithms(True)
    return SEED

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def configure_logger(log_file, log_level=logging.INFO):
    # 创建一个日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
    # logging.basicConfig(level=log_level,  # 设置日志级别为INFO
    #                     format='%(asctime)s [%(levelname)s] %(message)s',
    #                     handlers=[
    #                     logging.FileHandler(log_file),  # 将日志写入文件
    #                     logging.StreamHandler()  # 将日志打印到控制台
    #                     ])

    # 创建一个文件处理程序
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.INFO)

    # 创建一个格式化器
    # formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    # file_handler.setFormatter(formatter)

    # 添加处理程序到日志记录器
    # logger.addHandler(file_handler)

    # return logger