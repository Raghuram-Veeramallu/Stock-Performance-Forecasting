import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class HuggingFaceSummarizationBaseModel(object):

    def __init__(self, model_id) -> None:
        self.model_id = model_id
        if torch.cuda.is_available():
            print('Running on GPU!!')
            self.device = "cuda"
        else:
            self.device = "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        except OSError:
            raise ModuleNotFoundError(f'Model with name: {self.model_id} does not exists in Huggingface library. Please check the model_id before trying again.')
        
        self.tokenizer#.to(self.device)
        self.model#.to(self.device)
    
    def summarize(self):
        raise NotImplementedError('Summarize function not implemented')

    def encode_single(self, text_to_encode, return_tensors = 'pt', padding = True, truncation = True, max_length = 1024):
        return self.tokenizer.encode_plus(
            text_to_encode[0].to(self.device),
            return_tensors = return_tensors,
            padding = padding,
            truncation = truncation,
            max_length = max_length,
        )

    def decode_single(self, tokens, skip_special_tokens = True):
        return self.tokenizer.decode(tokens[0], skip_special_tokens=skip_special_tokens).to("cpu")

    def encode_batch(self, text_to_encode: list, return_tensors: str = 'pt', padding: bool = True, truncation: bool = True, max_length: int = 1024):

        return self.tokenizer.batch_encode_plus(
            text_to_encode.to(self.device),
            return_tensors = return_tensors,
            padding = padding,
            truncation = truncation,
            max_length = max_length,
        )

    def decode_batch(self, tokens, skip_special_tokens = True):
        return self.tokenizer.batch_decode(tokens[0], skip_special_tokens=skip_special_tokens).to("cpu")

    def generate_summaries(self, tokens, min_length: int = 10, max_length: int = 1024):
        return self.model.generate(
            tokens['input_ids'],
            min_length = min_length,
            max_length = max_length,
        )
