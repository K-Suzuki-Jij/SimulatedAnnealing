from simulated_annealing.integer.sa_system import SASystem
import pytest


def test_sa_system_1():
    interaction = {
        (0, 1): 1.0,
        (1, 2): -2.0,
        (2, 0): 3.0,
        (1, 1): -4.0,
        (0,): -0.1,
        (1,): 0.2,
        (2,): 0.3,
    }
    range_dict = {1: (0, 3), 2: (-1, 3)}
    sa_system = SASystem(interaction, range_dict, seed=0)

    assert sa_system.size == 3
    assert sa_system.J_list == [
        [(1, 1.0), (2, 3.0)],  # 0
        [(0, 1.0), (2, -2.0)],  # 1
        [(0, 3.0), (1, -2.0)],  # 2
    ]
    assert sa_system.h_list == [-0.1, 0.2, 0.3]
    assert sa_system.self_J_list == [0, -4.0, 0]
    assert sa_system.var_list[0].value in range(0, 2)
    assert sa_system.var_list[1].value in range(0, 4)
    assert sa_system.var_list[2].value in range(-1, 4)
    assert sa_system.index_list == [0, 1, 2]
    assert sa_system.index_map == {0: 0, 1: 1, 2: 2}

    new_state = [v.value for v in sa_system.var_list]
    for v in range(0, 2):
        new_state[0] = v
        dE = sa_system.calc_energy(new_state) - sa_system.calc_energy()
        assert sa_system.get_dE(0, v) == pytest.approx(dE)

    new_state[0] = sa_system.var_list[0].value
    for v in range(0, 3):
        new_state[1] = v
        dE = sa_system.calc_energy(new_state) - sa_system.calc_energy()
        assert sa_system.get_dE(1, v) == pytest.approx(dE)

    new_state[1] = sa_system.var_list[1].value
    for v in range(-1, 4):
        new_state[2] = v
        dE = sa_system.calc_energy(new_state) - sa_system.calc_energy()
        assert sa_system.get_dE(2, v) == pytest.approx(dE)

    sa_system.set_value(1, sa_system.var_list[1].get_candidate_value())

    new_state = [v.value for v in sa_system.var_list]
    for v in range(0, 2):
        new_state[0] = v
        dE = sa_system.calc_energy(new_state) - sa_system.calc_energy()
        assert sa_system.get_dE(0, v) == pytest.approx(dE)

    new_state[0] = sa_system.var_list[0].value
    for v in range(0, 3):
        new_state[1] = v
        dE = sa_system.calc_energy(new_state) - sa_system.calc_energy()
        assert sa_system.get_dE(1, v) == pytest.approx(dE)

    new_state[1] = sa_system.var_list[1].value
    for v in range(-1, 4):
        new_state[2] = v
        dE = sa_system.calc_energy(new_state) - sa_system.calc_energy()
        assert sa_system.get_dE(2, v) == pytest.approx(dE)


def test_sa_system_2():
    interaction = {
        ("a", "b"): 1.0,
        ("b", "c"): -2.0,
        ("c", "a"): 3.0,
        ("b", "b"): -4.0,
        ("a",): -0.1,
        ("b",): 0.2,
        ("c",): 0.3,
    }
    range_dict = {"b": (0, 3), "c": (-1, 3)}
    sa_system = SASystem(interaction, range_dict, seed=0)

    assert sa_system.size == 3
    assert sa_system.J_list == [
        [(1, 1.0), (2, 3.0)],  # 0
        [(0, 1.0), (2, -2.0)],  # 1
        [(0, 3.0), (1, -2.0)],  # 2
    ]
    assert sa_system.h_list == [-0.1, 0.2, 0.3]
    assert sa_system.self_J_list == [0, -4.0, 0]
    assert sa_system.var_list[0].value in range(0, 2)
    assert sa_system.var_list[1].value in range(0, 4)
    assert sa_system.var_list[2].value in range(-1, 4)
    assert sa_system.index_list == ["a", "b", "c"]
    assert sa_system.index_map == {"a": 0, "b": 1, "c": 2}


def test_sa_system_3():
    interaction = {
        ("a", "b"): 1.0,
        ("b", "c"): -1.0,
        ("c", "b"): -1.0,
        ("c", "a"): 3.0,
        ("b", "b"): -4.0,
        ("a",): -0.1,
        ("b",): 0.2,
        ("c",): 0.3,
    }
    range_dict = {"b": (0, 3), "c": (-1, 3)}
    sa_system_1 = SASystem(interaction, range_dict, seed=0)

    interaction = {
        ("a", "b"): 1.0,
        ("b", "c"): -2.0,
        ("c", "a"): 3.0,
        ("b", "b"): -4.0,
        ("a",): -0.1,
        ("b",): 0.2,
        ("c",): 0.3,
    }
    sa_system_2 = SASystem(interaction, range_dict, seed=0)

    assert sa_system_1.size == sa_system_2.size
    assert sa_system_1.J_list == sa_system_2.J_list
    assert sa_system_1.h_list == sa_system_2.h_list
    assert sa_system_1.self_J_list == sa_system_2.self_J_list
    assert sa_system_1.var_list[0].value == sa_system_2.var_list[0].value
    assert sa_system_1.var_list[1].value == sa_system_2.var_list[1].value
    assert sa_system_1.var_list[2].value == sa_system_2.var_list[2].value
    assert sa_system_1.index_list == sa_system_2.index_list
    assert sa_system_1.index_map == sa_system_2.index_map
    assert sa_system_1.dE_list == sa_system_2.dE_list
