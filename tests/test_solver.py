from simulated_annealing.integer.solver import solve


def test_integer_solver():
    interaction = {
        ("x", "y"): -3,
        ("x", "x"): 2,
        ("y", "y"): 2,
        ("x",): -4,
        ("y",): 5,
    }

    result = solve(
        interaction=interaction,
        range_dict={"x": (-2, 3), "y": (0, 4)},
        num_sweeps=50,
        num_samples=10,
        state_updater="HEAT_BATH",
        seed=0,
    )

    min_result = result.get_min_solution()
    for i in range(len(min_result)):
        assert min_result.solution_list[i] == {"x": 1, "y": 0}
        assert min_result.energy_list[i] == -2.0
        assert min_result.solving_time_list[i] > 0.0
        assert min_result.meta_data_list[i]["num_sweeps"] > 0
        assert min_result.meta_data_list[i]["seed"] >= 0
        assert min_result.meta_data_list[i]["T_min"] > 0.0
        assert min_result.meta_data_list[i]["T_max"] > 0.0
        assert "state_updater" in min_result.meta_data_list[i]
