import os
import random
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from a2c_ppo_acktr.datasets.utkinect_constants import UTKINECT_ACTIONS


def _normalize_label(label: str) -> str:
    return label.strip().replace(" ", "").lower()


class UTKinectDatasetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        history_window: int = 6,
        frame_skip: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.history_window = max(1, history_window)
        self.frame_skip = max(1, frame_skip)
        self._rng = random.Random(seed)
        self.action_space = spaces.Discrete(len(UTKINECT_ACTIONS))
        self._sequences = self._load_sequences()
        if not self._sequences:
            raise RuntimeError(f"No UTKinect sequences found under {dataset_root}")
        first_sequence = next(iter(self._sequences.values()))
        first_frame = self._read_image(first_sequence[0]["image_path"])
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=first_frame.shape,
            dtype=np.uint8,
        )
        self._sequence_ids = list(self._sequences.keys())
        self._current_sequence_id: Optional[str] = None
        self._current_index = 0
        self._current_info: Optional[Dict] = None
        self._current_sequence: List[Dict] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng.seed(seed)
        self._select_sequence()
        obs = self._current_frame()
        return obs, self.get_current_info()

    def step(self, action):
        self._current_index += self.frame_skip
        terminated = False
        if self._current_index >= len(self._current_sequence) - 1:
            terminated = True
            self._current_index = len(self._current_sequence) - 2
        obs = self._current_frame()
        info = self.get_current_info()
        reward = 0.0
        truncated = False
        return obs, reward, terminated, truncated, info

    def get_current_info(self):
        return dict(self._current_info) if self._current_info else {}

    def _select_sequence(self):
        seq_id = self._rng.choice(self._sequence_ids)
        sequence = self._sequences[seq_id]
        if len(sequence) < 2:
            self._select_sequence()
            return
        max_start = max(0, len(sequence) - self.frame_skip - 1)
        self._current_index = self._rng.randint(0, max_start)
        self._current_sequence_id = seq_id
        self._current_sequence = sequence
        self._current_info = self._build_info()

    def _current_frame(self):
        entry = self._current_sequence[self._current_index]
        self._current_info = self._build_info()
        return self._read_image(entry["image_path"])

    def _build_info(self):
        history_end = self._current_index
        history_start = max(0, history_end - self.history_window + 1)
        history = [
            self._current_sequence[i]["label"]
            for i in range(history_start, history_end + 1)
        ]
        next_label = self._current_sequence[self._current_index + 1]["label"]
        return {
            "sequence_id": self._current_sequence_id,
            "frame_index": self._current_sequence[self._current_index]["frame_index"],
            "action_history": history,
            "current_action": self._current_sequence[self._current_index]["label"],
            "target_next_action": next_label,
        }

    def _load_sequences(self) -> Dict[str, List[Dict]]:
        split_file = os.path.join(
            self.dataset_root, "splits", f"{self.split}_split.txt"
        )
        if not os.path.exists(split_file):
            raise FileNotFoundError(split_file)
        sequences: Dict[str, List[Dict]] = {}
        with open(split_file, "r") as f:
            sequence_files = [line.strip() for line in f if line.strip()]
        for seq_file in sequence_files:
            entries = self._parse_sequence(seq_file)
            if len(entries) < 2:
                continue
            seq_id = os.path.splitext(seq_file)[0]
            sequences[seq_id] = entries
        return sequences

    def _parse_sequence(self, seq_file: str) -> List[Dict]:
        gt_path = os.path.join(self.dataset_root, "groundTruth", seq_file)
        if not os.path.exists(gt_path):
            return []
        entries: List[Dict] = []
        with open(gt_path, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 2:
                    continue
                image_path, label = parts[0], parts[1]
                label = _normalize_label(label)
                if label == "undefined" or label not in UTKINECT_ACTIONS:
                    continue
                resolved_path = self._resolve_path(image_path)
                if resolved_path is None:
                    continue
                entries.append(
                    {
                        "image_path": resolved_path,
                        "label": label,
                        "frame_index": self._frame_index_from_path(resolved_path),
                    }
                )
        return entries

    def _resolve_path(self, raw_path: str) -> Optional[str]:
        raw_path = raw_path.strip()
        if os.path.exists(raw_path):
            return raw_path
        anchor = "utkinect/"
        lower = raw_path.lower()
        if anchor in lower:
            idx = lower.rfind(anchor) + len(anchor)
            candidate = os.path.join(self.dataset_root, raw_path[idx:])
            if os.path.exists(candidate):
                return candidate
        candidate = os.path.join(self.dataset_root, raw_path.split("/")[-1])
        if os.path.exists(candidate):
            return candidate
        # final attempt that joins RGB prefix
        rgb_idx = lower.rfind("rgb/")
        if rgb_idx != -1:
            rgb_relative = raw_path[rgb_idx:]
            candidate = os.path.join(self.dataset_root, rgb_relative)
            if os.path.exists(candidate):
                return candidate
        return None

    def _frame_index_from_path(self, path: str) -> int:
        basename = os.path.basename(path)
        digits = "".join(ch for ch in basename if ch.isdigit())
        return int(digits) if digits else -1

    def _read_image(self, path: str) -> np.ndarray:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))
