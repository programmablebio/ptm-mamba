import logging


class TrainLogger:
    def __init__(self, log_file="logfile.log"):
        logging.basicConfig(
            filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s"
        )

    def log(self, data):
        data = {k: round(v, 4) if isinstance(v, float) else v for k, v in data.items()}
        logging.info(data)
