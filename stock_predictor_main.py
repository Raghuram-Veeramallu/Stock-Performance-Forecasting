import os
import sys
import time

from pipeline import Pipeline

class StockPredictorMainRunner:

    def __init__(self) -> None:
        self.config_file = sys.argv[1]

        # to avoid warnings about parallelism in tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # os.environ["MASTER_ADDR"] = "127.0.0.1"

if __name__ == '__main__':
    start_time = time.time()
    stock_predictor_main = StockPredictorMainRunner()
    pipeline = Pipeline(stock_predictor_main.config_file)
    pipeline.run_pipeline()
    print('Total time taken for the whole process: {}'.format(time.time() - start_time))
