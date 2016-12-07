#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from reservoir.skeleton import fib

__author__ = "1Reinier"
__copyright__ = "1Reinier"
__license__ = "none"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
