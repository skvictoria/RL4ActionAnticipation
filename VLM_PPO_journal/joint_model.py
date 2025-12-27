import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence

# Add mmaam to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mmaam'))

from model.mMant import FUTR
from utils import read_mapping_dict, normalize_duration

class JointFUTR:
    def __init__(self, device, dataset_root, model_path=None, hidden_dim=2048, lr=1e-5): # LR lowered
        self.device = device
        self.dataset_root = dataset_root
        
        # Paths
        self.mapping_file = os.path.join(dataset_root, 'mapping_l2_changed.txt')
        self.features_path = os.path.join(dataset_root, 'features_img') 
        self.depth_features_path = os.path.join(dataset_root, 'features_depth')
        self.gt_path = os.path.join(dataset_root, 'groundTruth')
        
        # Labels and Mapping
        self.actions_dict = read_mapping_dict(self.mapping_file)
        self.n_class = len(self.actions_dict)# + 1
        self.pad_idx = self.n_class# + 1
        print("padding index: ", self.pad_idx)
        self.inverse_dict = {v: k for k, v in self.actions_dict.items()}
        #self.inverse_dict[self.n_class - 1] = "NONE"

        # Model Args
        self.args = SimpleNamespace(
            input_dim=2048,
            input_type='i3d_transcript',
            seg=True,
            anticipate=True,
            max_pos_len=512, # Reduced max len for stability (was 2000)
            n_query=8, 
            n_head=8,
            n_encoder_layer=6,
            n_decoder_layer=6
        )
        
        # Init Model
        self.model = FUTR(
            self.n_class, hidden_dim, src_pad_idx=self.pad_idx, device=device, args=self.args,
            n_query=self.args.n_query, n_head=self.args.n_head,
            num_encoder_layers=self.args.n_encoder_layer, num_decoder_layers=self.args.n_decoder_layer
        ).to(device)
        
        # Load Pretrained Weights
        if model_path and os.path.exists(model_path):
            print(f"[JointFUTR] Loading weights from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                # 'module.' 접두사 제거
                key = k.replace('module.', '')
                
                # [중요] 기존의 'context_projector' 필터링 로직 제거
                # 이제 파일 안에 해당 키가 있으면 불러오고, 없으면 건너뜁니다.
                new_state_dict[key] = v
            
            # strict=False로 설정하여, 
            # 1) Base 모델 로딩 시: context_projector 키가 없어도 에러 없이 넘어감 (랜덤 초기화 유지)
            # 2) Joint 모델 로딩 시: 모든 키를 정상적으로 로드함
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"[JointFUTR] Weights loaded. Missing keys (expected for Base model): {msg.missing_keys}")
        else:
            print("[JointFUTR] No checkpoint found or provided. Training from scratch.")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion_cls = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.criterion_reg = nn.MSELoss(reduction='none')
        self.sample_rate = 1

    def _seq2transcript(self, seq):
        """BaseDataset의 seq2transcript 구현체"""
        if not seq: return [], []
        transcript_action, transcript_dur = [], []
        action = self._norm(seq[0])
        transcript_action.append(self.actions_dict.get(action, self.pad_idx))
        last_i = 0
        for i in range(len(seq)):
            curr_action = self._norm(seq[i])
            if action != curr_action:
                action = curr_action
                transcript_action.append(self.actions_dict.get(action, self.pad_idx))
                transcript_dur.append((i - last_i) / len(seq))
                last_i = i
        transcript_dur.append((len(seq) - last_i) / len(seq))
        return transcript_action, transcript_dur

    def _prepare_batch(self, infos, training=False):
        batch_inputs, batch_targets, valid_indices = [], [], []
        batch_act_targets, batch_dur_targets = [], []

        for i, info in enumerate(infos):
            seq_id = info.get('sequence_id')
            if not seq_id: continue
            
            # 1. 전체 피처 로드하여 총 길이 확인
            feat_file = os.path.join(self.features_path, f"{seq_id}.npy")
            if not os.path.exists(feat_file): continue
            feats = np.load(feat_file)
            total_len = len(feats)

            # 2. [수정] obs_perc 결정 (BaseDataset 로직 반영)
            if training:
                # 학습 시 20%, 30%, 50% 중 하나를 무작위 선택하여 샘플링
                obs_perc = np.random.choice([0.2, 0.3, 0.5])
                observed_len = int(obs_perc * total_len)
            else:
                # 추론 시에는 RL 에이전트의 현재 진행 프레임 사용
                observed_len = info.get('frame_index', 0) + 1
            
            observed_len = max(1, min(observed_len, total_len))
            
            # 입력 피처 슬라이싱
            feats_seq = feats[:observed_len][::self.sample_rate]
            if len(feats_seq) > self.args.max_pos_len:
                feats_seq = feats_seq[-self.args.max_pos_len:]

            # 3. [수정] Ground Truth 로드 및 타겟 생성
            gt_file = os.path.join(self.gt_path, f"{seq_id}.txt")
            if not os.path.exists(gt_file): continue
            
            with open(gt_file, 'r') as f:
                # 'frame_index, action_label, ...' 형식 가정
                lines = [line.strip().split(',') for line in f.readlines() if ',' in line]
            all_labels = [l[1] for l in lines]
            
            # 입력 구간(Past)과 예측 구간(Future) 분리
            past_labels = all_labels[:observed_len][::self.sample_rate]
            # 미래 구간은 BaseDataset처럼 전체의 50% 정도를 타겟으로 삼음
            future_labels = all_labels[observed_len : observed_len + int(0.5 * total_len)][::self.sample_rate]

            # Segmentation 타겟 (Past)
            target_indices = [self.actions_dict.get(self._norm(lbl), self.pad_idx) for lbl in past_labels]
            full_target = torch.full((len(feats_seq),), self.pad_idx, dtype=torch.long)
            filled_len = min(len(feats_seq), len(target_indices))
            if filled_len > 0:
                full_target[-filled_len:] = torch.tensor(target_indices[-filled_len:]).long()

            # Action/Duration 타겟 (Future) - seq2transcript 로직 적용
            trans_act, trans_dur = self._seq2transcript(future_labels)
            
            act_target = torch.full((self.args.n_query,), self.pad_idx, dtype=torch.long)
            dur_target = torch.full((self.args.n_query,), 0.0, dtype=torch.float)
            
            q_len = min(self.args.n_query, len(trans_act))
            if q_len > 0:
                act_target[:q_len] = torch.tensor(trans_act[:q_len]).long()
                dur_target[:q_len] = torch.tensor(trans_dur[:q_len]).float()

            batch_inputs.append(torch.tensor(feats_seq).float())
            batch_targets.append(full_target)
            batch_act_targets.append(act_target)
            batch_dur_targets.append(dur_target)
            valid_indices.append(i)

        if not batch_inputs: return None, None, None, None, []

        padded_inputs = pad_sequence(batch_inputs, batch_first=True).to(self.device)
        padded_targets = pad_sequence(batch_targets, batch_first=True, padding_value=self.pad_idx).to(self.device)
        padded_act = torch.stack(batch_act_targets).to(self.device)
        padded_dur = torch.stack(batch_dur_targets).to(self.device)
            
        return padded_inputs, padded_targets, padded_act, padded_dur, valid_indices

    def _norm(self, lbl):
        return lbl.strip().replace(" ", "").lower()

    def predict_coarse(self, infos):
        """
        FUTR 모델의 Segmentation 출력을 사용하여 과거 액션 히스토리를 예측합니다.
        VLM의 프롬프트 구성을 위한 Coarse Label을 생성하는 데 사용됩니다.
        """
        self.model.eval()
        
        # 1. [수정] _prepare_batch에서 5개의 인자를 받음
        inputs, targets_seg, targets_act, targets_dur, valid_indices = self._prepare_batch(infos, training=False)
        
        # 데이터가 없는 경우 빈 히스토리 리스트 반환
        if inputs is None: 
            return [[] for _ in range(len(infos))]

        with torch.no_grad():
            # 2. 모델 추론 (test 모드에서는 targets_seg 정보를 사용하지 않거나 None으로 전달 가능)
            # FUTR 모델의 인터페이스에 따라 inputs만 넣거나 (inputs, None) 형태로 전달
            outputs = self.model(inputs, query=None, context=None, mode='test')
        
        # 3. Segmentation 결과 추출 [B, S, n_class] -> 인덱스 변환 [B, S]
        seg_logits = outputs['seg']
        seg_preds = seg_logits.max(-1)[1].cpu().numpy()
        
        # 결과를 저장할 리스트 초기화 (참조 무결성을 위해 리스트 컴프리헨션 사용)
        result_histories = [[] for _ in range(len(infos))]
        
        batch_idx = 0
        for i in range(len(infos)):
            if i in valid_indices:
                # 해당 배치의 예측 인덱스 시퀀스
                p_seq = seg_preds[batch_idx]
                
                # 4. [수정] inverse_dict를 사용하여 인덱스를 문자열로 변환
                # 이전에 actions_dict에서 'none'을 제외했으므로, 유효한 클래스만 결과에 포함됨
                hist_str = [
                    self.inverse_dict[p] for p in p_seq 
                    if p in self.inverse_dict
                ]
                
                # 중복되는 연속 액션을 하나로 합치고 싶다면 여기서 추가 처리가 가능합니다.
                # 현재는 프레임 단위의 모든 예측 결과를 리스트로 반환합니다.
                result_histories[i] = hist_str
                batch_idx += 1
            else:
                # 유효한 피처가 없는 경우 기존 info에 있는 히스토리 유지
                raw_hist = infos[i].get('action_history', [])
                result_histories[i] = [h for h in raw_hist if h and h.lower() != "none"]
                
        return result_histories

    def train_step(self, infos, fg_embedding):
        self.model.train()
        #inputs, targets, valid_indices = self._prepare_batch(infos, training=True)
        inputs, targets_seg, targets_act, targets_dur, valid_indices = self._prepare_batch(infos, training=True)
        
        if inputs is None:
            return 0.0

        valid_fg_embed = None
        if fg_embedding is not None:
            valid_fg_list = [fg_embedding[i] for i in valid_indices]
            if valid_fg_list:
                valid_fg_embed = torch.stack(valid_fg_list).to(self.device)

        self.optimizer.zero_grad()
        
        outputs = self.model((inputs, targets_seg), query=None, context=valid_fg_embed, mode='train')
        
        # 1. [Segmentation Loss]
        loss_seg = self.criterion_cls(outputs['seg'].view(-1, self.n_class), targets_seg.view(-1))
        
        # 2. Action Anticipation Loss
        if outputs['action'] is not None:
            loss_act = self.criterion_cls(outputs['action'].view(-1, self.n_class), targets_act.view(-1))
        else:
            loss_act = 0.0
        
        # 3. Duration Loss
        if outputs['duration'] is not None:
            output_dur = outputs['duration']
            dur_mask = (targets_dur > 0).float()
            output_dur = normalize_duration(output_dur, dur_mask)
            loss_dur = torch.sum(self.criterion_reg(output_dur, targets_dur) * dur_mask) / (torch.sum(dur_mask) + 1e-6)
        else:
            loss_dur = 0.0

        total_loss = loss_seg + loss_act + loss_dur
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[JointFUTR] Saved model to {path}")

    def predict_future(self, infos, fg_embedding):
        self.model.eval()
        inputs, _, valid_indices = self._prepare_batch(infos, training=False)
        
        if inputs is None: return []

        # 1. Fine-grained Embedding 준비
        valid_fg_embed = None
        if fg_embedding is not None:
            # 배치 순서에 맞게 정렬
            valid_fg_list = [fg_embedding[i] for i in valid_indices]
            if valid_fg_list:
                valid_fg_embed = torch.stack(valid_fg_list).to(self.device)

        with torch.no_grad():
            # 2. Context(fg_embed)를 넣어서 추론
            outputs = self.model(inputs, query=None, context=valid_fg_embed, mode='test')
            
        # 3. 결과 파싱 (Next Action)
        # outputs['action']: [B, n_query, C] -> 우리는 첫 번째 쿼리(바로 다음 행동)만 필요
        act_logits = outputs['action'] 
        act_preds = act_logits.max(-1)[1].cpu().numpy() # [B, n_query]
        
        result_future = ["UNDEFINED"] * len(infos)
        
        batch_idx = 0
        for i in range(len(infos)):
            if i in valid_indices:
                # 첫 번째 쿼리(0번 인덱스)가 바로 다음 행동 예측값
                n_act = act_preds[batch_idx][0] 
                pred_str = self.inverse_dict.get(n_act.item(), "UNDEFINED")
                result_future[i] = pred_str
                batch_idx += 1
                
        return result_future