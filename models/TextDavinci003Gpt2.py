
from .Gpt2SummarizationBase import Gpt2SummarizationBase

class TextDavinci003Gpt2(Gpt2SummarizationBase):

    def __init__(self, max_length: int = 1024, temperature: float = 0.1) -> None:
        super().__init__('text-davinci-003', max_length, temperature)
