import os
from typing import Optional, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class UtkinectSequenceDataset:
    def __init__(self, root, split="train"):
        """
        root: UTKinect 데이터셋 루트 폴더
        split: train / val / test
        """
        # TODO: 여기에 실제로 시퀀스 목록을 읽어오는 코드 작성
        # 예: self.sequences = [np.ndarray(T, H, W, 3), ...]
        #     self.labels    = [int, int, ...]
        self.sequences: List[np.ndarray] = ...
        self.labels: List[int] = ...
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        # frames: (T, H, W, 3)  또는 (T, feat_dim)
        frames = self.sequences[idx]
        label = self.labels[idx]
        return frames, label


class UtkinectsEnv(gym.Env):
    """
    UTKinect 시퀀스를 이용한 RL 환경
    - 한 에피소드 = 하나의 시퀀스
    - action = 클래스 예측 (Discrete(num_classes))
    - obs = 현재 프레임 (pixel 또는 skeleton feature)
    """
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 15,  # 대충
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        render_mode: Optional[str] = None,
        is_pixel: bool = True,
        max_steps: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = UtkinectSequenceDataset(root, split)
        self.num_classes = self.dataset.num_classes
        self.is_pixel = is_pixel
        self.render_mode = render_mode
        self.max_steps = max_steps

        # 예: 픽셀 기반 환경으로 할 경우 (H, W, C) 지정
        # 실제 UTKinect 전처리 결과에 맞게 수정 필요
        if self.is_pixel:
            H, W, C = 224, 224, 3  # 예시
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(H, W, C), dtype=np.uint8
            )
        else:
            feat_dim = 3 * 20  # joint 20개 * (x,y,z) 같은 구조라면
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(feat_dim,), dtype=np.float32
            )

        self.action_space = spaces.Discrete(self.num_classes)

        # 에피소드 상태용 변수들
        self.current_seq_idx = -1
        self.current_frames = None   # shape: (T, ...)
        self.current_label = None    # int
        self.t = 0                  # 현재 step 인덱스
        self.T = 0                  # 현재 시퀀스 길이

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # 시퀀스 하나 샘플링
        self.current_seq_idx = int(self.np_random.integers(len(self.dataset)))
        self.current_frames, self.current_label = self.dataset[self.current_seq_idx]
        self.T = self.current_frames.shape[0]
        self.t = 0

        obs = self._get_obs()
        info = {
            "seq_idx": self.current_seq_idx,
            "label": int(self.current_label),
        }
        return obs, info

    def _get_obs(self):
        """
        현재 timestep의 observation 리턴
        """
        # 에피소드 끝나도 마지막 frame을 그대로 보여 주는 방식
        idx = min(self.t, self.T - 1)
        frame = self.current_frames[idx]

        if self.is_pixel:
            # 이미 (H, W, C) np.uint8이라면 그대로
            return frame.astype(np.uint8)
        else:
            # skeleton feature라면 flatten 등
            return frame.astype(np.float32)

    def step(self, action: int):
        assert self.action_space.contains(int(action))

        # 기본적으로 reward = 0, 마지막에만 +1/-1 주는 구조
        reward = 0.0
        terminated = False
        truncated = False

        # 한 스텝 진행: 다음 프레임으로 이동
        self.t += 1

        # 에피소드 종료 조건: 시퀀스 끝 or max_steps 도달
        if self.t >= self.T or (self.max_steps is not None and self.t >= self.max_steps):
            terminated = True
            # 마지막 step의 action을 정답과 비교해서 보상
            reward = 1.0 if int(action) == int(self.current_label) else -1.0

        obs = self._get_obs()
        info = {
            "seq_idx": self.current_seq_idx,
            "t": self.t,
            "label": int(self.current_label),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()
        # human 모드는 필요하면 구현
        return None
