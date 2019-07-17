import sys
import logging

logger = logging.getLogger(__name__)
_DEBUG = sys.gettrace() is not None
