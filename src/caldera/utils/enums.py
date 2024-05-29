from enum import IntEnum


class DevSet(IntEnum):
    RP1T = 0
    FALCON = 1

class TransformerSubLayers(IntEnum):
    QUERY = 0
    KEY = 1
    VALUE = 2
    O = 3
    UP = 4
    GATE = 5
    DOWN = 6