# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
import glob
from collections import defaultdict

# UTKinects 데이터셋의 10개 고유 액션 레이블 + UNDEFINED
ACTION_LIST = [
    "walk", "sitDown", "standUp", "pickUp", "carry", "throw",
    "push", "pull", "waveHands", "clapHands", "UNDEFINED"
]
ACTION_MAPPING = {i: name for i, name in enumerate(ACTION_LIST)}


class UtkinectsEnv(gym.Env):
    """UTKinects 데이터셋을 위한 커스텀 Gym 환경 (groundTruth 폴더 구조 반영)"""
    metadata = {'render.modes': ['human']}
    def seed(self, seed=None):
        np.random.seed(seed)

    def __init__(self, rgb_path='/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect/RGB', ground_truth_path='/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect/groundTruth', split='train'):
        super(UtkinectsEnv, self).__init__()

        self.rgb_path = rgb_path
        self.ground_truth_path = ground_truth_path
        
        # 정답 파일을 로드하여 비디오별 프레임/레이블 목록 생성
        self.episodes = self._load_annotations()

        # Train/Test 스플릿
        self.episode_keys = self._split_data(split)
        if not self.episode_keys:
            raise ValueError(f"'{split}' 스플릿에 대한 데이터를 로드할 수 없습니다. 정답 파일과 경로를 확인하세요.")

        self.current_episode_idx = -1
        self.current_frame_idx_in_episode = 0
        
        self.action_space = spaces.Discrete(len(ACTION_LIST))
        self.observation_space = spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)

    def _load_annotations(self):
        if not os.path.isdir(self.ground_truth_path):
            raise FileNotFoundError(f"Ground truth 경로({self.ground_truth_path})를 찾을 수 없습니다.")
        
        grouped_data = defaultdict(lambda: {'frames': [], 'labels': []})
        
        # groundTruth 폴더 내의 모든 .txt 파일을 순회
        annotation_files = glob.glob(os.path.join(self.ground_truth_path, 's*_e*.txt'))
        
        for ann_file in annotation_files:
            video_id = os.path.basename(ann_file).replace('.txt', '') # e.g., 's01_e01'
            
            with open(ann_file, 'r') as f:
                for line in f:
                    # 라인 형식이 '프레임번호 액션레이블' 이라고 가정 (e.g., '420 walk')
                    parts = line.strip().split()
                    
                    if len(parts) < 2:
                        continue
                    
                    frame_num_str = parts[0]
                    label = parts[1].replace(',', '')

                    if label in ACTION_LIST:
                        
                        #frame_path = os.path.join(self.rgb_path, video_id, f'colorImg{frame_num_str}.jpg')
                        frame_path = frame_num_str
                        grouped_data[video_id]['frames'].append(frame_path)
                        grouped_data[video_id]['labels'].append(label)

        return {k: v for k, v in grouped_data.items() if v['frames']}

    def _split_data(self, split):
        episode_keys = sorted(self.episodes.keys())
        np.random.shuffle(episode_keys)
        
        split_index = int(len(episode_keys) * 0.8)
        
        if split == 'train':
            keys = episode_keys[:split_index]
        else:
            keys = episode_keys[split_index:]
        
        print(f"'{split}', loaded {len(keys)} number of episodes (videos).")
        return keys

    def reset(self, seed=None, options=None):
        # gymnasium 표준에 따라, reset 함수 내부에서 시드 설정을 처리합니다.
        super().reset(seed=seed)

        self.current_episode_idx = (self.current_episode_idx + 1) % len(self.episode_keys)
        self.current_frame_idx_in_episode = 0
        
        episode_key = self.episode_keys[self.current_episode_idx]
        print(f"episode {self.current_episode_idx + 1}/{len(self.episode_keys)} ('{episode_key}') start.")

        observation = self._get_observation()
        info = {} # gymnasium은 (관측, 정보) 튜플을 반환해야 합니다.

        return observation, info

    def _get_observation(self):
        episode_key = self.episode_keys[self.current_episode_idx]
        episode_data = self.episodes[episode_key]
        
        if self.current_frame_idx_in_episode >= len(episode_data['frames']):
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

        frame_path = episode_data['frames'][self.current_frame_idx_in_episode]
        frame_path = frame_path.replace(',', '')
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: {frame_path} cannot load image.")
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        return img

    def step(self, action):
        # main.py에서 보상을 직접 계산하고 덮어쓰므로, 여기서는 더미(dummy) 값을 반환합니다.
        reward = 0
        
        # 다음 프레임으로 이동합니다.
        self.current_frame_idx_in_episode += 1
        
        # 에피소드의 끝인지 확인합니다.
        episode_key = self.episode_keys[self.current_episode_idx]
        done = self.current_frame_idx_in_episode >= len(self.episodes[episode_key]['frames'])

        # gymnasium 표준에 따라, terminated와 truncated를 분리하여 사용합니다.
        terminated = done
        truncated = False # 시간 제한(time limit)으로 끝나는 경우는 없으므로 False로 고정합니다.

        # 다음 관측(observation)을 가져옵니다.
        observation = self._get_observation()
        info = {} # 추가 정보

        # gymnasium 표준 반환 형식: (관측, 보상, 종료 여부, 잘림 여부, 정보)
        return observation, reward, terminated, truncated, info

    def render(self, mode='human', close=False):
        obs = self._get_observation()
        if mode == 'human':
            cv2.imshow('UtkinectsEnv', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)