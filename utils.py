import functools
import time

import pandas as pd
import numpy as np


def time_total(function):
    """ 
        Wrapper function to decorate other functions and get their running time 
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        function(*args, **kwargs)
        t1 = time.time()

        print(f'Function {function.__name__}() took {t1-t0} seconds to execute.')
    
    return wrapper


if __name__ ==  "main":
    " Execute this program "
    pass



