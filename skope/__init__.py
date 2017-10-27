# Version number
__version__ = "0.0.1"

# Was skope imported from setup.py?
try:
    __SKOPE_SETUP__
except NameError:
    __SKOPE_SETUP__ = False

if not __SKOPE_SETUP__:

    from .simulateK2target import Target
