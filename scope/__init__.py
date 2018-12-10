import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__
from .simulatetarget import *
from .batch import *

# Was scope imported from setup.py?
try:
    __SCOPE_SETUP__
except NameError:
    __SCOPE_SETUP__ = False

if not __SCOPE_SETUP__:

    from .simulatetarget import Target
