from .HuggingFaceSummarizationBase import HuggingFaceSummarizationBaseModel
from .HumanCenteredSummarizationPegasus import HumanCenteredSummarizationPegasus
from .Gpt2SummarizationBase import Gpt2SummarizationBase
from .TextDavinci003Gpt2 import TextDavinci003Gpt2


MODEL_MAPPING = {
    'human_centered_pegasus': HumanCenteredSummarizationPegasus,
    'gpt2_text_davinci_003': TextDavinci003Gpt2,
}
