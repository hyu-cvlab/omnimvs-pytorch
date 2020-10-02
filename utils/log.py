# utils.log
# 
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
import sys
import logging

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format = 
        '[%(levelname).1s%(asctime)s.%(msecs)03d %(process)d ' \
        '%(filename)s:%(lineno)d] %(message)s',
    datefmt='%m%d %H:%M:%S')
__logger = logging.getLogger()
LOG_INFO = __logger.info
LOG_ERROR = __logger.error
LOG_WARNING = __logger.warning
LOG_DEBUG = __logger.debug
LOG_CRITICAL = __logger.critical