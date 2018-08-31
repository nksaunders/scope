# Version number
__version__ = "1.0.3"

# Was scope imported from setup.py?
try:
    __SCOPE_SETUP__
except NameError:
    __SCOPE_SETUP__ = False

if not __SCOPE_SETUP__:

    from .simulatetarget import Target
