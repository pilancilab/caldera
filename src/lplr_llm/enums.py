from enum import Enum

class AlgorithmType(Enum):
    ALTERNATING_MIXED_LPLR = 0
    DIRECT_SVD_LPLR = 1
    LOFTQ = 2
    LOFTQ_LPLR = 3

class ADMMType(Enum):
    ADMM_Q = 0
    ADMM_R = 1
    ADMM_S = 2