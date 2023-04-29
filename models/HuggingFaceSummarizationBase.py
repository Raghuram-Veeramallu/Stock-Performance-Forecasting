import os

import deepspeed
import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.deepspeed import HfDeepSpeedConfig

class HuggingFaceSummarizationBaseModel(object):

    def __init__(self, model_id) -> None:
        self.model_id = model_id

        # configuring the device
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # TODO: for now testing
        # if self.device == 'cuda':
        # initializing deep speed
        deepspeed.init_distributed()

        # distributed setup
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))

        try:
            self.model_config = AutoConfig.from_pretrained(self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        except OSError:
            raise ModuleNotFoundError(f'Model with name: {self.model_id} does not exists in Huggingface library. Please check the model_id before trying again.')

        model_hidden_size = self.model_config.d_model

        train_batch_size = 1 * world_size

        # initializing deep speed with huggingface
        ds_config = self.__deep_speed_config(model_hidden_size, train_batch_size)
        ds_hf_config = HfDeepSpeedConfig(ds_config) # to keep this object alive

        # initialise Deepspeed ZeRO and store only the engine object
        self.ds_engine = deepspeed.initialize(model=self.model, config_params=ds_config)[0]
        self.ds_engine.module.eval()  # inference
    
    def __deep_speed_config(self, model_hidden_size, train_batch_size):
        return {
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": False
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": model_hidden_size * model_hidden_size,
                "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
                "stage3_param_persistence_threshold": 10 * model_hidden_size
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }

    def summarize(self):
        raise NotImplementedError('Summarize function not implemented')

    def encode_single(self, text_to_encode, return_tensors = 'pt', padding = True, truncation = True, max_length = 1024):
        return self.tokenizer.encode_plus(
            text_to_encode[0],
            return_tensors = return_tensors,
            padding = padding,
            truncation = truncation,
            max_length = max_length,
        ).to(device=self.local_rank)

    def decode_single(self, tokens, skip_special_tokens = True):
        return self.tokenizer.decode(tokens[0], skip_special_tokens=skip_special_tokens)

    def encode_batch(self, text_to_encode: list, return_tensors: str = 'pt', padding: bool = True, truncation: bool = True, max_length: int = 1024):

        return self.tokenizer.batch_encode_plus(
            text_to_encode,
            return_tensors = return_tensors,
            padding = padding,
            truncation = truncation,
            max_length = max_length,
        ).to(device=self.local_rank)

    def decode_batch(self, tokens, skip_special_tokens = True):
        return self.tokenizer.batch_decode(tokens[0], skip_special_tokens=skip_special_tokens)

    def generate_summaries(self, tokens, min_length: int = 10, max_length: int = 1024):
        with torch.no_grad():
            generated_text = self.ds_engine.module.generate(
                tokens['input_ids'],
                min_length = min_length,
                max_length = max_length,
                synced_gpus=True,
            )
        return generated_text
        # return self.model.generate(
        #     tokens['input_ids'],
        #     min_length = min_length,
        #     max_length = max_length,
        # )
