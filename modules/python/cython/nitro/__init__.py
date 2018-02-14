import os
if 'NITF_PLUGIN_PATH' not in os.environ:
    from pkg_resources import resource_filename
    os.environ['NITF_PLUGIN_PATH'] = resource_filename(__name__, 'plugins')
from .dataextension_segment import *
from .field import *
from .error import *
from .nitro import *
from .tre import TRE
