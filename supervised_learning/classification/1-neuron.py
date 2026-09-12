#!/usr/bin/env python3
"""Defines a Neuron performing binary classification"""
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Class constructor

        Args:
            nx (int): the number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output"""
        return self.__A
