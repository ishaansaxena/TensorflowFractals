import numpy as np

# Color Profiles

def RGB(a_cyclic):
    return [
        200 * np.cos(a_cyclic),
        200 * np.sin(a_cyclic),
        - 155 * np.cos(a_cyclic)
    ]

def PURPLE(a_cyclic):
    return [
        80 - 50 * np.cos(a_cyclic),
        0 + 1 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)
    ]

# Global Profiles

DEFAULT_PROFILE = RGB
COLOR_PROFILE = DEFAULT_PROFILE

# Set profile to function

def SetColorProfile(func):
    global COLOR_PROFILE
    COLOR_PROFILE = func
