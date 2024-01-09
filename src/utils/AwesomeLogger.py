import os
import sys
import logging

class Logger(object):

    def __init__(self,
                logger_name = 'Logger',
                address = '',
                level = logging.DEBUG,
                console_level = logging.INFO,
                file_level = logging.INFO,
                mode = 'a',
                **params):
        super(Logger, self).__init__()
        
        self.logger_name = logger_name

        logging.basicConfig()

        self.address = address
        
        self.instance = logging.getLogger(logger_name)
        self.instance.setLevel(level)
        self.instance.propagate = False

        formatter = logging.Formatter('%(levelname)s: %(message)s\n')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.instance.addHandler(console_handler)
        
        file_handler = logging.FileHandler(address, mode = mode)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        self.instance.addHandler(file_handler)
    
    def _correct_message(self, message):
        output = f"{message}"
        return output

    def debug(self, message):
        self.instance.debug(self._correct_message(message))

    def info(self, message):
        self.instance.info(self._correct_message(message))

    def warning(self, message):
        self.instance.warning(self._correct_message(message))

    def error(self, message):
        self._handle_errors(message)
        self.instance.error(self._correct_message(message))

    def critical(self, message, slack_channel = None):
        self.instance.critical(self._correct_message(message))

def submit_loggers(enums):

    logger = Logger(
        logger_name = "simulation_logger",
        address=os.path.join(enums.TRADE_REPORTS_DIR, "Log.log"),
    )

    logger = Logger(
        logger_name = "research_logger",
        console_level = logging.INFO,
        address=os.path.join(enums.TRADE_REPORTS_DIR, "ResearchLog.log"),
    )