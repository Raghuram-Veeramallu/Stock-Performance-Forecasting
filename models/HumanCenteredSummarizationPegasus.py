from .HuggingFaceSummarizationBase import HuggingFaceSummarizationBaseModel

class HumanCenteredSummarizationPegasus(HuggingFaceSummarizationBaseModel):

    def __init__(self) -> None:
        self.model_id = 'human-centered-summarization/financial-summarization-pegasus'
        super().__init__(self.model_id)
