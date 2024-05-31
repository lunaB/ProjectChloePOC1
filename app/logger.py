import logging
import contextlib
import warnings
warnings.filterwarnings("ignore")


log_format = '[ %(levelname)s ]:\t%(message)s'
logging.basicConfig(format=log_format, level=logging.ERROR)

def get_logger(name: str):
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  return logger
  
@contextlib.contextmanager
def wait_log(logger: logging.Logger, message):
  try:
    yield
    logger.log(logging.INFO, f'{message} [done]')
  except Exception:
    logger.log(logging.WARNING, f'{message} [fail]')
    raise


if __name__ == "__main__":
  logger = get_logger(__name__)
  logger.error("Logger initialized1")
  logger.info("Logger initialized1")
  logger.warning("Logger initialized2")
  logger.debug("Logger initialized3")