import logging
from time import time

class Logging(object):
    INSTANCE = None

    def __init__(self, config):
        if Logging.INSTANCE is not None:
            raise ValueError("An Logging Instance already exists")

        self.logger = logging.getLogger()

        logging.basicConfig(filename='LOGGER', level = logging.DEBUG)
        fileHandler = logging.FileHandler(config.LOGGER_PATH + '/msg.log')
        streamHandler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)

    @classmethod
    def get_instance(cls, config):
        if cls.INSTANCE is None:
            cls.INSTANCE = Logging(config)
        return cls.INSTANCE

    @staticmethod
    def training_loss_logger(step, start_time, loss, config):
        if step%config.LOG_PERIOD == 0:
            Logging.get_instance(config).logger.info("STEP [{}] :: time {} loss : {}".format(step, time() - start_time, loss))

    @staticmethod
    def validation_loss_logger(step, start_time, loss, config):
        Logging.get_instance(config).logger.info("VALIDATION {}:: time {} loss : {}".format(step, time() - start_time, loss))
