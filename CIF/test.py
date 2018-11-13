from CIF.logging.logging import Logging as lh
from CIF.config import *
from time import time
import signal

config = Config("const_net_info.yml")
logger = lh.get_instance(config).logger

logger.info("Logging Test")
GLOBAL_START_TIME = time()


pid = os.getpid()
iter =0
try:
    while True:
        iter = iter+1

except(KeyboardInterrupt, SystemExit, Exception, signal.SIGINT, signal.SIGKILL, os.kill(pid, signal.SIGINT)):
    logger.info("Interrupt")

except:
    raise


finally:
    logger.info("Finally")


for i in range(10):
    lh.training_loss_logger(i, GLOBAL_START_TIME, i+1, config)