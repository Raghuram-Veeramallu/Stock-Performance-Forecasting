import sqlite3
import pandas as pd
import configparser

class DatabaseConnector:

    def __init__(self) -> None:
        self.__get_connection_details()
        self.cursor = self.create_connection(self.__db_file_path)

    # get connection details from .cfg file
    def __get_connection_details(self) -> None:
        self.__config_parser = configparser.ConfigParser()
        self.__config_parser.read('../environ.cfg')
        self.__db_file_path = self.__config_parser.get('DATABASE', 'FILE_PATH', fallback='')
        if self.__db_file_path == '':
            raise ConnectionError('Database filepath required to establish connection.')

    # function to create a sqlite3 connection
    def create_connection(self, conn_string):
        connection = sqlite3.connect(conn_string)
        # create a connection cursor
        cursor = connection.cursor()
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

    def generate_query(self, table_name, query_type='read_all', columns=None, condition=''):
    #     if query_type == 'colnames':
    #         query = f'PRAGMA table_info('{table_name}');'
        if (query_type == 'read') and (columns != None):
            query = f'SELECT {columns} FROM {table_name}{condition};'
        else:# query_type == 'read_all':
            query = f'SELECT * FROM {table_name}{condition};'
        return query
