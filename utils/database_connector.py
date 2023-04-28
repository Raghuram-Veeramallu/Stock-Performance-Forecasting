import sqlite3
import pandas as pd
import configparser

class DatabaseConnector:

    __CREATE_PREDICTIONS_METADATA_SQL = './utils/sql/create_metadata_table.sql'
    __CREATE_PREDICTIONS_TABLE = './utils/sql/create_predicitons_table.sql'
    __INSERT_INTO_PREDICTIONS_TABLE = './utils/sql/insert_into_metadata_table.sql'

    def __init__(self) -> None:
        self.connection = None
        self.__db_file_path = self.__get_connection_details()
        self.cursor = self.create_connection(self.__db_file_path)

    # get connection details from .cfg file
    def __get_connection_details(self) -> None:
        self.__config_parser = configparser.ConfigParser()
        self.__config_parser.read('./environ.cfg', encoding='utf-8')
        __db_file_path = self.__config_parser.get('DATABASE', 'FILE_PATH', fallback="")
        if __db_file_path == '':
            raise ConnectionError('Database filepath required to establish connection.')
        return __db_file_path

    # function to create a sqlite3 connection
    def create_connection(self, conn_string):
        # TODO: conn_string not working. Had to manually replace this.
        # self.connnection = sqlite3.connect(f"{conn_string}")
        self.connection = sqlite3.connect("/Users/harisairaghuramveeramallu/earning_transcripts.db")
        # create a connection cursor
        cursor = self.connection.cursor()
        return cursor

    # execute a query
    def execute_query(self, query, return_type=list):
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        if return_type == pd.DataFrame:
            column_names = [description[0] for description in self.cursor.description]
            return pd.DataFrame(results, columns=column_names)
        if return_type == list:
            if len(results) > 0:
                if len(results[0]) == 1:
                    return list(map(lambda x: x[0], results))
                else:
                    return list(map(lambda x: list(x), results))        
        return None

    def generate_query(self, table_name=None, query_type='read_all', columns=None, condition=''):
    #     if query_type == 'colnames':
    #         query = f'PRAGMA table_info('{table_name}');'
        if (query_type == 'read') and (columns != None):
            query = f'SELECT {columns} FROM {table_name}{condition};'
        elif query_type == 'create_pred_table':
            with open(self.__CREATE_PREDICTIONS_TABLE, 'r') as f:
                query = f.read()
            query = query.format(table_name)
        elif query_type == 'create_pred_metadata_table':
            with open(self.__CREATE_PREDICTIONS_METADATA_SQL, 'r') as f:
                query = f.read()
        elif query_type == 'max_id_metadata':
            query = 'SELECT MAX(id) FROM predictions_metadata;'
        else:# query_type == 'read_all':
            query = f'SELECT * FROM {table_name}{condition} LIMIT 1;'
        return query

    def insert_into_metadata_table(self, table_name, method_name, author, summarization_model, classification_model, date):
        with open(self.__INSERT_INTO_PREDICTIONS_TABLE, 'r') as f:
            query = f.read()
        query = query.format(
            table_name = table_name,
            method_name = method_name,
            author = author,
            summarization_model = summarization_model,
            classification_model = classification_model,
            current_date = date,
        )
        return self.execute_query(query)

    def save_df_to_db(self, df, table_name, if_exists='replace', index=False):
        df.to_sql(table_name, con=self.connection, if_exists=if_exists, index=index)

    def __del__(self):
        if self.connection:
            self.connection.close()
