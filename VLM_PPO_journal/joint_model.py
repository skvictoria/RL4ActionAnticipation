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
from utils import read_mapping_dict

class JointFUTR:
    def __init__(self, device, dataset_root, model_path=None, hidden_dim=2048, lr=1e-5): # LR lowered
        self.device = device
        self.dataset_root = dataset_root
        
        # Paths
        self.mapping_file = os.path.join(dataset_root, 'mapping_l2_changed.txt')
        self.features_path = os.path.join(dataset_root, 'features_img') 
        self.depth_features_path = os.path.join(dataset_root, 'features_depth')
        
        # Labels and Mapping
        self.actions_dict = read_mapping_dict(self.mapping_file)
        self.n_class = len(self.actions_dict) + 1
        self.pad_idx = self.n_class + 1
        self.inverse_dict = {v: k for k, v in self.actions_dict.items()}
        self.inverse_dict[self.n_class - 1] = "NONE"

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
        self.sample_rate = 1

    def _prepare_batch(self, infos, training=False):
        batch_inputs = []
        batch_targets = []
        valid_indices = []

        for i, info in enumerate(infos):
            if 'sequence_id' not in info: continue
            
            seq_id = info['sequence_id']
            curr_frame = info['frame_index']
            
            feat_file = os.path.join(self.features_path, f"{seq_id}.npy")
            if not os.path.exists(feat_file): continue
            
            feats = np.load(feat_file)
            limit = curr_frame + 1
            
            # [FIX] Slice only the last max_pos_len frames to avoid super long sequences
            feats_seq = feats[:limit][::self.sample_rate]
            if len(feats_seq) > self.args.max_pos_len:
                feats_seq = feats_seq[-self.args.max_pos_len:]
            
            batch_inputs.append(torch.tensor(feats_seq).float())
            valid_indices.append(i)
            
            if training:
                # [FIX] Better Target Alignment
                gt_hist = info.get('action_history', []) # e.g., length 6
                target_indices = [self.actions_dict.get(self._norm(lbl), self.n_class-1) for lbl in gt_hist]
                
                input_len = len(feats_seq)
                target_len = len(target_indices)
                
                # Initialize target tensor with PAD_IDX (Ignore Index)
                full_target = torch.full((input_len,), self.pad_idx, dtype=torch.long)
                
                # Fill the END of the target with the available ground truth
                # Assume gt_hist corresponds to the *last* frames
                if target_len > 0:
                    start_idx = max(0, input_len - target_len)
                    # Handle case where gt is longer than input (unlikely due to max_pos_len logic, but safe slice)
                    gt_slice = target_indices[-min(input_len, target_len):]
                    full_target[start_idx:] = torch.tensor(gt_slice).long()
                
                batch_targets.append(full_target)

        if not batch_inputs: return None, None, []

        padded_inputs = pad_sequence(batch_inputs, batch_first=True).to(self.device)
        padded_targets = None
        if training and batch_targets:
            # Padding value for batch should also be pad_idx
            padded_targets = pad_sequence(batch_targets, batch_first=True, padding_value=self.pad_idx).to(self.device)
            
        return padded_inputs, padded_targets, valid_indices

    def _norm(self, lbl):
        return lbl.strip().replace(" ", "").lower()

    def predict_coarse(self, infos):
        self.model.eval()
        inputs, _, valid_indices = self._prepare_batch(infos, training=False)
        
        if inputs is None: return []

        with torch.no_grad():
            outputs = self.model(inputs, query=None, context=None, mode='test')
            
        seg_preds = outputs['seg'].max(-1)[1].cpu().numpy()
        result_histories = [[]] * len(infos)
        
        batch_idx = 0
        for i in range(len(infos)):
            if i in valid_indices:
                p_seq = seg_preds[batch_idx]
                
                # [DEBUG] Print Raw Prediction Stats once in a while or if empty
                #unique, counts = np.unique(p_seq, return_counts=True)
                #print(f"DEBUG Preds Frame {i}: {dict(zip(unique, counts))}")
                
                hist_str = [self.inverse_dict.get(p, "UNDEFINED") for p in p_seq]
                
                # [FIX] Filter NONE but handle empty result
                filtered_hist = [h for h in hist_str if h != "NONE"]
                
                # If filtered is empty, it means model predicted all NONE. 
                # This suggests collapse. Return raw for debugging or fallback.
                if not filtered_hist and len(hist_str) > 0:
                    # Fallback: Just take last 5 raw predictions even if NONE to see what happened
                    # Or keep it empty, but log it.
                    pass 

                result_histories[i] = filtered_hist
                batch_idx += 1
            else:
                result_histories[i] = infos[i].get('action_history', [])
                
        return result_histories

    def train_step(self, infos, fg_embedding):
        self.model.train()
        inputs, targets, valid_indices = self._prepare_batch(infos, training=True)
        
        if inputs is None or targets is None:
            return 0.0

        valid_fg_embed = None
        if fg_embedding is not None:
            valid_fg_list = [fg_embedding[i] for i in valid_indices]
            if valid_fg_list:
                valid_fg_embed = torch.stack(valid_fg_list).to(self.device)

        self.optimizer.zero_grad()
        
        outputs = self.model((inputs, targets), query=None, context=valid_fg_embed, mode='train')
        
        # Segmentation Loss
        seg_logits = outputs['seg'].view(-1, self.n_class)
        seg_targets = targets.view(-1)
        
        # Only compute loss where targets != pad_idx
        loss_seg = self.criterion_cls(seg_logits, seg_targets)
        
        loss_act = 0
        if 'action' in outputs:
            next_acts = []
            valid_next_targets = []
            for i, idx in enumerate(valid_indices):
                tgt_str = infos[idx].get('target_next_action', 'undefined')
                # Check if target is valid
                if tgt_str and tgt_str.lower() != 'undefined':
                    tgt_idx = self.actions_dict.get(self._norm(tgt_str), self.n_class-1)
                    next_acts.append(i) # Keep track of batch index
                    valid_next_targets.append(tgt_idx)
            
            if next_acts:
                next_targets = torch.tensor(valid_next_targets).long().to(self.device)
                # Select only valid batch items
                pred_next = outputs['action'][next_acts, 0, :] 
                loss_act = self.criterion_cls(pred_next, next_targets)
        
        total_loss = loss_seg + loss_act
        total_loss.backward()
        
        # [FIX] Gradient Clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
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