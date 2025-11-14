 
"""
nnenum: Neural Network Verification Tool
"""

# Define which public symbols users get when they do:
#     from nnenum import *
__all__ = []

# Optionally expose top-level functionality gradually like:
# from .nnenum import some_function
# __all__.append("some_function")


def get_version():
    """Return the package version."""
    try:
        from importlib.metadata import version
        return version("nnenum")
    except Exception:
        return "unknown"
