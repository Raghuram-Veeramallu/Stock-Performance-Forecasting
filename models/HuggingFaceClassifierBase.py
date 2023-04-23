import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class HuggingFaceClassifierBase(object):

    TASK = "sentiment-analysis"

    # def __init__(self, model_id, num_labels, id2label, label2id) -> None:
    def __init__(self, model_id, kwargs={}) -> None:
        self.model_id = model_id
        # self.num_labels = num_labels
        # self.id2label = id2label
        # self.label2id = label2id
        try:
            # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # self.model = AutoModelForSequenceClassification.from_pretrained(
            #     model_id, self.num_labels, id2label, label2id,
            # )
            self.classifier_pipeline = pipeline(
                self.TASK, model = self.model_id, **kwargs
            )
        except OSError:
            raise ModuleNotFoundError(f'Model with name: {self.model_id} does not exists in Huggingface library. Please check the model_id before trying again.')

    def pipeline_text_classification(self, text):
        pipeline_out = self.classifier_pipeline(text)
        return pipeline_out

    def tokenize_text(self, text):
        return self.tokenizer(text, return_tensors='pt')
    
    def obtain_classification_results(self, inputs):
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_label = logits.argmax().item()
        return self.id2label[predicted_label]