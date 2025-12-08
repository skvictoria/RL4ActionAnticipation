import torch
import numpy as np
import os
import sys
from types import SimpleNamespace

# Add mmaam to sys.path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'mmaam'))

from model.mMant import FUTR
from utils import read_mapping_dict

class CoarsePredictor:
    def __init__(self, device, dataset_root, model_path, hidden_dim=2048):
        self.device = device
        self.dataset_root = dataset_root
        
        # Paths setup
        self.mapping_file = os.path.join(dataset_root, 'mapping_l2_changed.txt')
        self.features_path = os.path.join(dataset_root, 'features_img') # Or features_img_gaussian/noise_030
        self.depth_features_path = os.path.join(dataset_root, 'features_depth')
        
        # Load dictionary
        self.actions_dict = read_mapping_dict(self.mapping_file)
        self.n_class = len(self.actions_dict) + 1
        self.pad_idx = self.n_class + 1
        self.inverse_dict = {v: k for k, v in self.actions_dict.items()}
        self.inverse_dict[self.n_class - 1] = "NONE" # Assuming last class is NONE

        # Args for FUTR model (Needs to match training config)
        self.args = SimpleNamespace(
            input_dim=2048, # Feature dim
            input_type='i3d_transcript', # or 'gt'
            seg=True,
            anticipate=True,
            max_pos_len=2000,
            n_query=8, 
            n_head=8,
            n_encoder_layer=6, # Default
            n_decoder_layer=6  # Default
        )
        
        # Init Model
        self.model = FUTR(
            self.n_class, hidden_dim, src_pad_idx=self.pad_idx, device=device, args=self.args,
            n_query=self.args.n_query, n_head=self.args.n_head,
            num_encoder_layers=self.args.n_encoder_layer, num_decoder_layers=self.args.n_decoder_layer
        ).to(device)
        
        # Load Checkpoint
        if os.path.exists(model_path):
            print(f"Loading FUTR predictor from {model_path}")
            # Handling DataParallel state dict if necessary
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
        else:
            print(f"Warning: FUTR model path {model_path} not found. Predictions will be random initialized.")
            
        self.model.eval()
        self.sample_rate = 1 # Set appropriately, e.g. 10 if training used 10

    def predict_batch(self, infos):
        """
        infos: List of dicts from envs.step(). Contains 'sequence_id' and 'frame_index'.
        Returns: 
            predicted_histories: List[List[str]] (Predicted Coarse Labels up to current frame)
            predicted_next_actions: List[str] (Anticipated Next Coarse Label)
        """
        batch_inputs = []
        batch_depths = []
        valid_indices = []
        
        # Prepare Batch
        for i, info in enumerate(infos):
            if 'sequence_id' not in info:
                continue
                
            seq_id = info['sequence_id']
            # frame_index is the CURRENT frame index. We need features up to this point.
            # Note: Features might be subsampled.
            curr_frame = info['frame_index']
            
            # Load Features
            feat_file = os.path.join(self.features_path, f"{seq_id}.npy")
            depth_file = os.path.join(self.depth_features_path, f"{seq_id}.npy")
            
            if not os.path.exists(feat_file) or not os.path.exists(depth_file):
                continue

            feats = np.load(feat_file)
            depths = np.load(depth_file)
            
            # Slice up to current frame (approximate mapping)
            # Assuming 1-to-1 mapping or matching sampling. 
            # If utkinects features are extracted per frame, just slice.
            # If features are subsampled during extraction, need to adjust index.
            # Typically UTKinect features might be per-frame or per-segment. Assuming per-frame aligned for now.
            # Applying sample rate as used in mmaam prediction
            
            # Length check
            seq_len = min(len(feats), len(depths))
            limit = min(curr_frame + 1, seq_len) # +1 to include current
            
            feats_seq = feats[:limit][::self.sample_rate]
            depths_seq = depths[:limit][::self.sample_rate]
            
            batch_inputs.append(torch.tensor(feats_seq).float())
            batch_depths.append(torch.tensor(depths_seq).float())
            valid_indices.append(i)

        if not batch_inputs:
            # Fallback if no valid features found
            return [info.get('action_history', []) for info in infos], ["UNDEFINED"] * len(infos)

        # Pad sequences
        from torch.nn.utils.rnn import pad_sequence
        padded_inputs = pad_sequence(batch_inputs, batch_first=True).to(self.device)
        padded_depths = pad_sequence(batch_depths, batch_first=True).to(self.device)
        
        # Run Inference
        with torch.no_grad():
            # forward(inputs, query, mode='test', ...) -> inputs=RGB, depth passed via modification or wrapper
            # Wait, mmaam FUTR forward signature: forward(self, inputs, query, mode='train', ...)
            # But predict_utkinects.py calls: model(inputs=..., depth_features=...)
            # We need to make sure the model class supports depth_features in forward.
            # Looking at mMant.py provided: forward(self, inputs, query, ...)
            # It DOES NOT seem to take depth_features in the provided mMant.py text!
            # However, predict_utkinects.py passes it. This suggests the mMant.py provided might be incomplete or 
            # predict_utkinects.py is using a modified version.
            # Based on provided mMant.py, it only takes `inputs`. 
            # I will pass `inputs` only for now to match the class definition provided, 
            # but if the actual model requires depth, you must update mMant.py.
            # Assuming provided mMant.py is correct:
            
            # The provided mMant.py forward: forward(self, inputs, query, mode='train', epoch=0, idx=0)
            # It seems it doesn't use depth.
            
            outputs = self.model(padded_inputs, query=None, mode='test')

        # Decode Outputs
        # outputs['seg']: [B, T, C] -> History
        # outputs['action']: [B, C] -> Next Action
        
        seg_logits = outputs['seg']
        act_logits = outputs['action']
        
        pred_histories = []
        pred_next = []
        
        seg_preds = seg_logits.max(-1)[1].cpu().numpy()
        act_preds = act_logits.max(-1)[1].cpu().numpy()
        
        batch_idx = 0
        result_histories = [[]] * len(infos)
        result_next = [""] * len(infos)

        for i in range(len(infos)):
            if i in valid_indices:
                # Decode history
                p_seq = seg_preds[batch_idx]
                # Filter out padding if necessary or just take valid length
                valid_len = len(batch_inputs[batch_idx])
                p_seq = p_seq[:valid_len]
                
                hist_str = [self.inverse_dict.get(p, "UNDEFINED") for p in p_seq]
                # Filter UNDEFINED or NONE if needed
                hist_str = [h for h in hist_str if h != "NONE"]
                
                # Decode next
                n_act = act_preds[batch_idx]
                next_str = self.inverse_dict.get(n_act, "UNDEFINED")
                
                result_histories[i] = hist_str
                result_next[i] = next_str
                batch_idx += 1
            else:
                # Fallback
                result_histories[i] = infos[i].get('action_history', [])
                result_next[i] = "UNDEFINED"
                
        return result_histories, result_next