""" Activations

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F
import tensorflow as tf


def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return tf.math.multiply(x, tf.nn.sigmoid(x))

class Swish(tf.keras.layers.Layer):
    def __init__(self, inplace: bool=False):
        super(Swish, self).__init__()
        self.inplace = inplace
    def call(self, x):
        return swish(x, self.inplace)



def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return tf.math.multiply(x, tf.math.tanh(tf.math.softplus(x)))

class Mish(tf.keras.layers.Layer):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def call(self, x):
        return mish(x)


def sigmoid(x, inplace: bool = False):
    return tf.nn.sigmoid(x)


# PyTorch has this, but not with a consistent inplace argmument interface
class Sigmoid(tf.keras.layers.Layer):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def call(self, x):
        return sigmoid(x, self.inplace)


def tanh(x, inplace: bool = False):
    return tf.nn.tanh()


class Tanh(tf.keras.layers.Layer):
    def __init__(self, inplace: bool = False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def call(self, x):
        return tanh(x, self.inplace)


def hard_swish(x, inplace: bool = False):
    inner = F.relu6(x + 3.).div_(6.)
    inner = tf.math.scalar_mul(1./6., tf.nn.relu6(x + 3.))
    return tf.math.multiply(inner, x)


class HardSwish(tf.keras.layers.Layer):
    def __init__(self, inplace: bool = False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def call(self, x):
        return hard_swish(x, self.inplace)


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def call(self, x):
        return hard_sigmoid(x, self.inplace)


def hard_mish(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return tf.scalar_mul(0.5, tf.math.multiply(x, tf.clip_by_value(x + 2.0, 0.0, 2.0)))

class HardMish(tf.keras.layers.Layer):
    def __init__(self, inplace: bool = False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    def call(self, x):
        return hard_mish(x, self.inplace)


class PReLU(tf.keras.layers.Layer):
    """Applies PReLU (w/ dummy inplace arg)
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25, inplace: bool = False) -> None:
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)
        self.prelu = tf.keras.layers.PReLU()
    def call(self, x):
        return self.prelu(x)


def gelu(x, inplace: bool = False):
    return tf.nn.gelu(x)


class GELU(tf.keras.layers.Layer):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super(GELU, self).__init__()

    def call(self, x):
        return gelu(x)
