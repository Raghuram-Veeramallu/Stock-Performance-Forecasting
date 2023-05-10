from .HuggingFaceSummarizationBase import HuggingFaceSummarizationBaseModel

class HumanCenteredSummarizationPegasus(HuggingFaceSummarizationBaseModel):

    def __init__(self) -> None:
        self.model_id = 'human-centered-summarization/financial-summarization-pegasus'
        super().__init__(self.model_id)
    
    def summarize(self, config, transcripts):
        max_length = config['summarization']['max_length']
        truncate = config['summarization']['truncate']
        padding = config['summarization']['padding']
        return_tensors = config['summarization']['return_tensors']
        gen_min_length = config['summarization']['generation_min_length']
        gen_max_length = config['summarization']['generation_max_length']
        skip_special_tokens = config['summarization']['skip_special_tokens']

        # encode the data
        if len(transcripts) > 1:
            tokens = self.encode_batch(
                transcripts, 
                return_tensors = return_tensors, 
                padding = padding, 
                truncation = truncate, 
                max_length = max_length,
            )
        else:
            tokens = self.encode_single(
                transcripts, 
                return_tensors = return_tensors, 
                padding = padding, 
                truncation = truncate, 
                max_length = max_length,
            )

        summarizations = self.generate_summaries(
            tokens, 
            min_length = gen_min_length, 
            max_length = gen_max_length,
        )

        if len(transcripts) > 1:
            response = self.decode_batch(
                summarizations,
                skip_special_tokens = skip_special_tokens
            )
        else:
            response = self.decode_single(
                summarizations,
                skip_special_tokens = skip_special_tokens
            )

        return response
