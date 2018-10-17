from .version import __version__
from .simulatetarget import *

# Was scope imported from setup.py?
try:
    __SCOPE_SETUP__
except NameError:
    __SCOPE_SETUP__ = False

if not __SCOPE_SETUP__:

    from .simulatetarget import Target
