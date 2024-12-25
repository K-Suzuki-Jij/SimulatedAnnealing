from simulated_annealing.integer.variable import Variable
import pytest


def test_variable_init():
    assert Variable(0, 10, 0).value in range(0, 11)
    assert Variable(3, 3, 0).value == 3

    with pytest.raises(ValueError):
        v = Variable(1, 0, 0)


def test_variable_candidate():
    v = Variable(0, 1, 0)
    assert v.get_candidate_value() in range(0, 2)
    assert v.get_candidate_value() != v.value
