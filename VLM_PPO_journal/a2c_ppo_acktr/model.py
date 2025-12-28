import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
import torch.nn.init as init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class VLMValue(nn.Module):
    """
    actually the base is also used for generation!
    """
    def __init__(self, base):
        super(VLMValue, self).__init__()
        self.base = base
        # 수정된 Value Head
        self.value_head = nn.Sequential(
            nn.LayerNorm(4096),        # [추가] LLM의 Outlier 값을 잡아주어 NaN 방지
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(base.device, dtype=torch.float32) # [수정] float16 -> float32

    def forward(self, input_ids, image_tensor):
        if image_tensor.size(0) != 1:
            input_ids = input_ids.broadcast_to(image_tensor.size(0), input_ids.size(-1))

        image_tensor = image_tensor.to(self.base.device, dtype=self.base.dtype)
        _, _, _, _, inputs_embeds, _ = self.base.prepare_inputs_labels_for_multimodal(
            input_ids.to(self.base.device), None, None, None, None, image_tensor)
        
        inputs_embeds = inputs_embeds.to(self.base.device, dtype=self.base.dtype)
        assert inputs_embeds.shape[1] > 256
        
        outputs = self.base(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True)
        
        # 마지막 레이어의 마지막 토큰 Hidden State를 가져옵니다.
        # 이 값은 현재 BFloat16 상태입니다.
        hidden_states = outputs.hidden_states[-1][:, -1]
        
        # [핵심 수정] BFloat16 데이터를 Float32로 캐스팅합니다.
        # value_head(LayerNorm 등)가 Float32로 설정되어 있으므로 타입을 맞춰줘야 합니다.
        with torch.cuda.amp.autocast(enabled=False):
            hidden_states = hidden_states.float()      
            values = self.value_head.float()(hidden_states)
        
        return values


class VLMPolicy(nn.Module):
    def __init__(self, tokenizer,
                image_processor,
                value_model,
                args,
                INPUT_IDS,
                projection_f,
                base_kwargs=None):
        """
        projection_f: the postprocessing function to parse text action
        """
        super(VLMPolicy, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.value_model = value_model
        self.base = value_model.base
        self.INPUT_IDS = INPUT_IDS
        self.projection_f = projection_f

    def process_obs(self, obs):
        #process the observation with the image processor
        processed_images = obs
        return self.image_processor.preprocess(processed_images, return_tensors='pt')['pixel_values'].to(dtype=self.base.dtype)

    def act(self, inputs, deterministic=False, INPUT_IDS=None):
        image_tensor = self.process_obs(inputs)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        value, output_ids, text_action, action_log_prob, action_tokens_log_prob = llava_generate(value_model = self.value_model,
                                                    tokenizer = self.tokenizer,
                                                    input_ids = INPUT_IDS,
                                                    image_tensor = image_tensor,
                                                    args = self.args)
        action = self.projection_f(text_action)
        return value, output_ids, action, action_log_prob, action_tokens_log_prob, text_action

    def get_value(self, inputs, INPUT_IDS=None):
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        image_tensor = self.process_obs(inputs)
        return self.value_model(input_ids = INPUT_IDS, image_tensor = image_tensor)

    def evaluate_actions(self, inputs, output_ids, INPUT_IDS=None):
        image_tensor = self.process_obs(inputs)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        value, action_log_prob, _ = llava_evaluate(value_model = self.value_model,
                                        input_ids = INPUT_IDS,
                                        output_ids = output_ids,
                                        image_tensor = image_tensor,
                                        temperature = self.args.temperature,
                                        thought_prob_coef = self.args.thought_prob_coef)
        return value, action_log_prob
