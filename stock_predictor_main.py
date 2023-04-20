import sys

from pipeline import Pipeline

class StockPredictorMainRunner:

    def __init__(self) -> None:
        self.config_file = sys.argv[1]

if __name__ == '__main__':
    stock_predictor_main = StockPredictorMainRunner()
    pipeline = Pipeline(stock_predictor_main.config_file)
    pipeline.run_pipeline()
