#!/usr/bin/env python3
"""
    Tensorflow project
"""
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activation=[]):
    """ creates forward propagation computation graph for NN """
    create_layer = __import__('1-create_layer').create_layer
    inp = x
    for i in range(len(layer_sizes)):
        inp = create_layer(inp, layer_sizes[i], activation[i])
    return inp
