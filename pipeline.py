import os
from datetime import date, datetime
import re
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd

from utils.database_connector import DatabaseConnector
import utils.file_load_store_utils as file_io_utils
import models

class Pipeline(object):

    __POSSIBLE_DATA_FETCHES = {
        'transcript', 'company', 'label', 'price'
    }

    def __init__(self, config_filepath: str) -> None:
        self.config = file_io_utils.load_configs_from_yaml(config_filepath)
        self.transcript_data  = self.__retrieve_data_from_database(data_type='transcript')

    def __retrieve_data_from_database(self, data_type='transcript', fetch_results_as=pd.DataFrame):
        if data_type in self.__POSSIBLE_DATA_FETCHES:
            database_connector = DatabaseConnector()
            if self.config['processing_config']['chunked']:
                offset = self.config['processing_config']['offset']
                limit = self.config['processing_config']['limit']
                condition = f'LIMIT {limit} OFFSET {offset}'
            else:
                condition = ''
            query = database_connector.generate_query(data_type, condition)
            response = database_connector.execute_query(query, return_type=fetch_results_as)
        else:
            raise Exception('Not a possible data fetch operation')
        
        return response
    
    def persist_data_to_database(self, dataframe: pd.DataFrame):
        database_connector = DatabaseConnector()
        # creating metadata table if not exists
        metadata_query = database_connector.generate_query(query_type='create_pred_metadata_table')
        _ = database_connector.execute_query(metadata_query)

        # retrieve the previous max index
        max_idx_query = database_connector.generate_query(query_type='max_id_metadata')
        id = database_connector.execute_query(max_idx_query)[0]
        id = id + 1 if id is not None else 1
        new_table_name = f'predictions_{str(id)}'

        # create an entry in metadata table and retrieve the id
        database_connector.insert_into_metadata_table(
            table_name = new_table_name,
            method_name = self.config['run_config']['method_name'],
            author = str(self.config['run_config']['author']),
            summarization_model = self.config['summarization']['model_name'],
            classification_model = self.config['classification']['model_name'],
            date = str(datetime.now()),
        )

        # persist data into database
        database_connector.save_df_to_db(dataframe, new_table_name, index=True)


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
    
    def run_summarization(self, transcripts):     
        # Run summarization model
        summarization_model_name = self.config['summarization']['model_name']
        summarizarion_model_class = models.SUMMARIZATION_MODEL_MAPPING[summarization_model_name]
        summarization_model = summarizarion_model_class()
        summarized_transcripts = []
        for each_transcript in tqdm(transcripts, desc="Running for each speaker"):
            summarized_transcripts.append(summarization_model.summarize(self.config, [each_transcript]))

        return summarized_transcripts

    def run_classification(self, summarized_transcripts, speakers):
        # Run classification model
        classifier_model_name = self.config['classification']['model_name']
        max_length = self.config['classification']['max_length']
        classifier_model_class = models.CLASSIFICATION_MODEL_MAPPING[classifier_model_name]
        classifier_model = classifier_model_class(max_length)
        predicted_labels = classifier_model.classify(self.config, summarized_transcripts)

        return predicted_labels

    def convert_summarization_results_to_df(self, data, transcripts, summarized_transcripts, speakers):
        reference_data = pd.DataFrame(data[['symbol', 'year', 'quarter', 'date']]).T
        if speakers is not None:
            summarized_transcripts_df = pd.DataFrame(list(zip(speakers, transcripts, summarized_transcripts)), columns=['speakers', 'original_transcript', 'summarized_transcript'])
        else:
            summarized_transcripts_df = pd.DataFrame(list(zip(transcripts, summarized_transcripts)), columns=['original_transcript', 'summarized_transcript'])

        summarized_df = pd.concat([reference_data, summarized_transcripts_df], axis=1)
        summarized_df[summarized_df.columns[:4]] = summarized_df[summarized_df.columns[:4]].ffill()
        return summarized_df

    def convert_classification_results_to_df(self, summarization_dataframe, prediction_results):
        prediction_df = pd.DataFrame(prediction_results)
        results_df = pd.concat([summarization_dataframe, prediction_df], axis = 1)
        return results_df

    def sequential_process_for_each_row(self, each_row: tuple):
        _, data = each_row

        run_only_summarization = self.config['run_config']['run_only_summarization']
        run_only_classification = self.config['run_config']['run_only_classification']

        if (not run_only_classification):
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

        if run_only_summarization:
            summarized_transcripts = self.run_summarization(transcripts)
            response = self.convert_summarization_results_to_df(data, transcripts, summarized_transcripts, speakers)
            return response
        elif run_only_classification:
            # retrieve the summarization results
            # summarized_transcripts = list(summarization_dataframe.groupby(['symbol', 'year', 'quarter', 'date'])['summarized_transcript'].apply(list))
            summarized_transcripts = data['summarized_transcript']
            speakers = data['speakers']
            summarization_dataframe = pd.DataFrame([data]).reset_index()
            summarization_dataframe.columns = ['symbol', 'year', 'quarter', 'date', 'speakers', 'summarized_transcript']
            summarization_dataframe = summarization_dataframe.explode(['speakers', 'summarized_transcript']).reset_index()
            predicted_labels = self.run_classification(summarized_transcripts, speakers)
            prediction_df = self.convert_classification_results_to_df(summarization_dataframe, predicted_labels)
            return prediction_df
        else:
            summarized_transcripts = self.run_summarization(transcripts)
            predicted_labels = self.run_classification(summarized_transcripts, speakers)
            summarized_df = self.convert_summarization_results_to_df(data, transcripts, summarized_transcripts, speakers)
            prediction_df = self.convert_classification_results_to_df(summarized_df, predicted_labels)
            return prediction_df

    def persist_summarization_results(self, results):
        file_io_utils.create_folder_if_not_exists(self.config['summarization']['summarization_output_filepath'])
        summarized_df = pd.concat(results, axis=0)
        summarization_model_name = self.config['summarization']['model_name']
        if self.config['processing_config']['chunked']:
            offset = self.config['processing_config']['offset']
            limit = self.config['processing_config']['limit']
            filename = f'{summarization_model_name}_summarized_{str(date.today())}_off_{offset}_lim_{limit}.csv'
        else:
            filename = f'{summarization_model_name}_summarized_{str(date.today())}.csv'
        filepth = os.path.join(self.config['summarization']['summarization_output_filepath'], filename)
        file_io_utils.export_data_frame_to_csv(summarized_df, filepth)

    def persist_classification_results(self, results):
        file_io_utils.create_folder_if_not_exists(self.config['classification']['classification_output_filepath'])
        prediction_df = pd.concat(results, axis=0)
        prediction_model_name = self.config['classification']['model_name']
        filename = f'{prediction_model_name}_classification_{str(date.today())}.csv'
        filepth = os.path.join(self.config['classification']['classification_output_filepath'], filename)
        # file persist
        file_io_utils.export_data_frame_to_csv(prediction_df, filepth)
        # db persist
        self.persist_data_to_database(prediction_df)

    def persist_outputs(self, results, forced_summarized_res=False):
        if forced_summarized_res:
            self.persist_summarization_results(results)
        else:
            if self.config['run_config']['run_only_summarization']:
                self.persist_summarization_results(results)
            elif self.config['run_config']['run_only_classification']:
                self.persist_classification_results(results)
            else:
                self.persist_classification_results(results)

    def run_pipeline(self):
        run_only_classification = self.config['run_config']['run_only_classification']

        if (not run_only_classification):
            if self.config['processing_config']['parallellize_runs']:
                # pool of processess
                pool = Pool()
                results = pool.map(self.sequential_process_for_each_row, self.transcript_data.iterrows())
            else:
                results = []
                for each_row in tqdm(self.transcript_data.iterrows(), desc="Running for each transcript"):
                    results.append(self.sequential_process_for_each_row(each_row))
        else:
            summarization_filepath = self.config['classification']['summarization_input_filepath']
            summarization_dataframe = file_io_utils.read_data_from_pd_dataframe(summarization_filepath)
            grouped_df = summarization_dataframe.groupby(['symbol', 'year', 'quarter', 'date'])[['speakers', 'summarized_transcript']].agg(list)
            if self.config['processing_config']['parallellize_runs']:
                # pool of processess
                pool = Pool()
                results = pool.map(self.sequential_process_for_each_row, grouped_df.iterrows())
            else:
                results = []
                for each_row in grouped_df.iterrows():
                    results.append(self.sequential_process_for_each_row(each_row))

        self.persist_outputs(results)
