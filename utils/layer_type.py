from enum import Enum


class LayerType(Enum):
    DROPOUT = "dropout"
    LAYER_NORM = "layer_norm"
    LINEAR = "linear"
    SUPER_BLOCK = "super_block"
    FORBIDDEN_BLOCK = "forbidden_block"
    AGGREGATOR = "aggregator"
    SUPER_LINEAR = 'super_linear'
    DECOMPOSER = "decomposer"
