
# python standard library
import os
import sys
from distutils.util import strtobool

# this package
from commoncode.index_builder import create_toctree

HTML_ONLY = strtobool(os.environ.get("HTML_ONLY", 'off'))
if HTML_ONLY:
   create_toctree()