import configparser

import openai

class Gpt2SummarizationBase(object):

    # not sure, need to tune this
    CHAT_GPT_MAX_TOKENS_LIMIT = 15097

    def __init__(self, model: str, max_tokens: int = 1024, temperature: float = 0.1) -> None:
        self.model = model
        self.__openai_organization_key = None
        self.__openai_api_key = None
        self.__get_secrets()
        # initialize the api key
        openai.organization = self.__openai_organization_key
        openai.api_key = self.__openai_api_key

        self.max_tokens = max_tokens
        self.temperature = temperature

        # if self.model is None:
        #     raise Exception('Model not set!!')

    # get secrets from .cfg file
    def __get_secrets(self) -> None:
        self.__config_parser = configparser.ConfigParser()
        self.__config_parser.read('../environ.cfg')
        self.__openai_organization_key = self.__config_parser.get('CHATGPT', 'OPENAI_ORGANIZATION')
        self.__openai_api_key = self.__config_parser.get('CHATGPT', 'OPENAI_API_KEY')

    # find a better way to batch generate summaries
    def summarize_text(self, text):
        if isinstance(text, str):
            text = [text]

        base_prompt = 'Summarize this text: \n "{text}"'

        summaries_generated = []

        for each_text_prompt in text:
            current_prompt = base_prompt.format(text = each_text_prompt[:self.CHAT_GPT_MAX_TOKENS_LIMIT])
            summaries_generated.append(
                self.prompt_chatgpt(current_prompt)
            )
        
        return summaries_generated

    
    def prompt_chatgpt(self, prompt: str):
        return openai.Completion.create(
            model = self.model,
            prompt = prompt,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
        )
