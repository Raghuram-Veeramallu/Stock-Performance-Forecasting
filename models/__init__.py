from .HuggingFaceSummarizationBase import HuggingFaceSummarizationBaseModel
from .HumanCenteredSummarizationPegasus import HumanCenteredSummarizationPegasus
from .Gpt2SummarizationBase import Gpt2SummarizationBase
from .TextDavinci003Gpt2 import TextDavinci003Gpt2

from .HuggingFaceClassifierBase import HuggingFaceClassifierBase
from .Dunn2BC22ClassifierDistilBert import Dunn2BC22ClassifierDistilBert


SUMMARIZATION_MODEL_MAPPING = {
    'human_centered_pegasus': HumanCenteredSummarizationPegasus,
    'gpt2_text_davinci_003': TextDavinci003Gpt2,
}

CLASSIFICATION_MODEL_MAPPING = {
    'dunn2BC22_distilbert': Dunn2BC22ClassifierDistilBert,
}
