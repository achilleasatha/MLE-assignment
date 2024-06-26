"""Example test module."""

import pytest


@pytest.mark.parametrize("inputs, expected", [(1, 2), (3, 4)])
@pytest.mark.skip(reason="Disable example tests")
def test_example(inputs, expected):
    """A sample test :param inputs: :param expected: :return:"""
    assert inputs + 1 == expected
