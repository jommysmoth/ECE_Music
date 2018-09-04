"""DOCSTRING."""
import logging


class Borg:
    """DOCSTRING."""

    _shared_state = {}

    def __init__(self):
        """DOCSTRING."""
        self.__dict__ = self._shared_state


class CfgLogger(Borg):
    """DOCSTRING."""

    def __init__(self, level=None):
        """DOCSTRING."""
        if level is None:
            self.logLevel = logging.INFO
        else:
            pass

    def create_logger(self, _name):
        """DOCSTRING."""
        logger = logging.getLogger(_name)
        logger.setLevel(self.logLevel)

        handler = logging.FileHandler('output_files/log_examp.log')
        handler.setLevel(self.logLevel)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        return logger
