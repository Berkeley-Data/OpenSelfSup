from openselfsup.utils import Registry

MODELS = Registry('model')
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
MEMORIES = Registry('memory')
LOSSES = Registry('loss')
INPUT_MODULES = Registry('input_module')