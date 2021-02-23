#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: time_metrics.py
@desc: All metric utils regarding time measures.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
from time import time


def timeit(function):
    """
    Time passed function as a decorator.
    """
    def timed(*args: list, **kwargs: dict):
        """
        Measure time of given function.

        :param args: List of arguments of given function.
        :param kwargs: Dictionary of arguments of given function.
        """
        start = time()
        result = function(*args, **kwargs)
        stop = time()
        return result, stop-start
    return timed
