# Library for statistical tools

import numpy as np

def mad_statistics(pop):

    med = np.median(pop)
    mad = np.median(np.abs(np.asarray(pop) - med))

    return mad
