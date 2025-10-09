import os
import numpy as np
import torch


# 새로 만든 커스텀 환경을 import
from breakfast_env import BreakfastEnv 

def make_env(env_id, seed, rank, log_dir, allow_early_resets, dataset_path):
    def _thunk():
        # env_id 대신 BreakfastEnv를 직접 사용
        env = BreakfastEnv(dataset_path=dataset_path)
        env.seed(seed + rank)
        # ... (이하 동일)
        return env
    return _thunk

def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  dataset_path): # dataset_path 인자 추가
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, dataset_path)
        for i in range(num_processes)
    ]
    # ... (이하 동일)



