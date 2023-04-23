import pandas as pd

from .HuggingFaceClassifierBase import HuggingFaceClassifierBase

class Dunn2BC22ClassifierDistilBert(HuggingFaceClassifierBase):

    def __init__(self, max_length = None) -> None:
        self.model_id = 'DunnBC22/distilbert-base-uncased-Financial_Sentiment_Analysis'
        kwargs = {}
        if max_length is not None:
            kwargs = {
                'max_length': max_length,
            }
        super().__init__(self.model_id, kwargs)

    def classify(self, config, transcripts):
        per_speaker_classification = config['classification']['per_speaker_classification']

        if not per_speaker_classification:
            combined_text = ' '.join(transcripts)
            response = self.pipeline_text_classification(combined_text)
        else:
            response = self.pipeline_text_classification(transcripts)
        
        return pd.DataFrame(response)
