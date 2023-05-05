import os
import re
from multiprocessing import Pool

import pandas as pd

from utils.database_connector import DatabaseConnector
import utils.file_load_store_utils as file_io_utils
import models

class Pipeline(object):

    __POSSIBLE_DATA_FETCHES = {
        'transcript', 'company'
    }

    def __init__(self, config_filepath: str) -> None:
        self.transcript_data  = self.__retrieve_data_from_database(data_type='transcript')
        self.config = file_io_utils.load_configs_from_yaml(config_filepath)

    def __retrieve_data_from_database(self, data_type='transcript', fetch_results_as=pd.DataFrame):
        if data_type in self.__POSSIBLE_DATA_FETCHES:
            database_connector = DatabaseConnector()
            query = database_connector.generate_query(data_type)
            response = database_connector.execute_query(query, return_type=fetch_results_as)
        else:
            raise Exception('Not a possible data fetch operation')
        
        return response

    # preliminary cleaning step before performing the pipelining steps
    def preliminary_cleaning_of_transcripts(self, data: pd.Series) -> pd.Series:
        # clean_spl_symbols_fn = lambda x: '.' + x.replace('\n', '')

        # apply the cleaning function to the transcript data
        # adding a '.' at the beginning for the speaker splitter to be able to work properly
        data['transcript'] = '.' + data['transcript'].replace('\n', '')

        return data

    # function to split the transcript based on speakers
    def split_transcript_based_on_speakers(self, transcript_text: str):
        # remove the trailing ':' from speaker names
        clean_speaker_names_fn = lambda x: x.strip(' ')[:-1]

        speakers = re.findall(r'(?<=[.?])\s*[A-Z][a-zA-Z]*(?:\s+[A-Za-z]\.)?(?:\s+[A-Za-z]+)*\s*:', transcript_text)
        each_speaker_transcript = re.split(r'(?<=[.?])\s*[A-Z][a-zA-Z]*(?:\s+[A-Za-z]\.)?(?:\s+[A-Za-z]+)*\s*:', transcript_text)

        # skipping the first sentence as it only contains '.'
        each_speaker_transcript = each_speaker_transcript[1:]

        assert len(speakers) == len(each_speaker_transcript)

        # cleaning speaker names
        speakers = list(map(clean_speaker_names_fn, speakers))

        return speakers, each_speaker_transcript

    def sequential_process_for_each_row(self, each_row: tuple):
        idx, data = each_row

        # clearning before processing the data
        data = self.preliminary_cleaning_of_transcripts(data)
        
        # split according to speakers
        if self.config['data_preprocessing_config']['run_per_speaker']:
            speakers, transcripts = self.split_transcript_based_on_speakers(data['transcript'])
            if self.config['data_preprocessing_config']['persist_per_speaker_transcripts']:
                dir = self.config['data_preprocessing_config']['per_speaker_transcript_file_path']
                file_name = f"{data['symbol']}_{data['year']}_{data['quarter']}.csv"
                file_io_utils.export_data_frame_to_csv(
                    pd.DataFrame(zip(speakers, transcripts), columns=['speakers', 'transcript']),
                    output_fp = os.path.join(dir, file_name),
                    keep_indices = False,
                )
        else:
            # else just use the transcripts as it is
            transcripts = data['transcript']
            speakers = None
        
        # Run summarization model
        model_name = self.config['summarization']['model_name']
        max_length = self.config['summarization']['max_length']
        truncate = self.config['summarization']['truncate']
        padding = self.config['summarization']['padding']
        return_tensors = self.config['summarization']['return_tensors']
        gen_min_length = self.config['summarization']['generation_min_length']
        gen_max_length = self.config['summarization']['generation_max_length']
        skip_special_tokens = self.config['summarization']['skip_special_tokens']

        model_class_name = models.MODEL_MAPPING[model_name]

        model = model_class_name()
        
        # encode the data
        if self.config['data_preprocessing_config']['run_per_speaker']:
            tokens = model.encode_batch(
                transcripts, 
                return_tensors = return_tensors, 
                padding = padding, 
                truncation = truncate, 
                max_length = max_length,
            )
        else:
            tokens = model.encode_single(
                transcripts, 
                return_tensors = return_tensors, 
                padding = padding, 
                truncation = truncate, 
                max_length = max_length,
            )

        summarizations = model.generate_summaries(
            tokens, 
            min_length = gen_min_length, 
            max_length = gen_max_length,
        )

        if self.config['data_preprocessing_config']['run_per_speaker']:
            response = model.decode_batch(
                summarizations,
                skip_special_tokens = skip_special_tokens
            )
        else:
            response = model.decode_single(
                summarizations,
                skip_special_tokens = skip_special_tokens
            )
        
        return response


    def run_pipeline(self):

        if self.config['processing_config']['parallellize_runs']:
            # pool of processess
            pool = Pool()
            results = pool.map(self.sequential_process_for_each_row, self.transcript_data.iterrows())
        else:
            results = []
            for each_row in self.transcript_data.iterrows():
                results.append(self.sequential_process_for_each_row(each_row))

        results_df = pd.concat([self.transcript_data, pd.DataFrame(results)], axis=1)
        results_df.to_csv('./final_response.csv', index = False)
