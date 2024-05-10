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

class DevSet(Enum):
    RP1T = 0
    FALCON = 1

class TransformerSubLayers(Enum):
    QUERY = 0
    KEY = 1
    VALUE = 2
    O = 3
    UP = 4
    GATE = 5
    DOWN = 6