from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from simulated_annealing.integer.state_generator import (
    StateGeneratorMetropolis,
    StateGeneratorHeatBath,
    StateGeneratorSuwaToda,
)
from simulated_annealing.integer.sa_system import SASystem

import time


@dataclass
class Results:
    solution_list: List[Dict[Any, int]]
    energy_list: List[float]
    solving_time_list: List[float]
    meta_data_list: List[Dict[str, Any]] = None

    def get_min_solution(self) -> "Results":
        min_energy = min(self.energy_list)
        min_index_list = [i for i, e in enumerate(self.energy_list) if e == min_energy]
        return Results(
            solution_list=[self.solution_list[i] for i in min_index_list],
            energy_list=[self.energy_list[i] for i in min_index_list],
            solving_time_list=[self.solving_time_list[i] for i in min_index_list],
            meta_data_list=self.meta_data_list,
        )

    def __len__(self):
        return len(self.solution_list)


def _solve(
    interaction: Dict[Tuple[Any, ...], float],
    range_dict: Dict[Any, Tuple[int, int]],
    num_sweeps: int,
    T_min: Optional[float] = None,
    T_max: Optional[float] = None,
    state_updater: str = "METROPOLIS",
    seed: Optional[int] = None,
) -> Tuple[Dict[Any, int], float, Dict[str, Any]]:
    sa_system = SASystem(interaction, range_dict, seed)

    if state_updater == "METROPOLIS":
        new_state_generator = StateGeneratorMetropolis()
    elif state_updater == "HEAT_BATH":
        max_num_state = max(v.num_states for v in sa_system.var_list)
        new_state_generator = StateGeneratorHeatBath(max_num_state)
    elif state_updater == "SUWA-TODO":
        max_num_state = max(v.num_states for v in sa_system.var_list)
        new_state_generator = StateGeneratorSuwaToda(max_num_state)
    else:
        raise ValueError("Invalid state updater")

    if T_min is None or T_max is None:
        temp_max, temp_min = sa_system.get_min_max_temperature()
        T_max = T_max if T_max is not None else temp_max
        T_min = T_min if T_min is not None else temp_min

    for sweeps in range(num_sweeps):
        T = (T_min / T_max) ** (sweeps / (num_sweeps - 1)) * T_max
        for i in range(sa_system.size):
            new_state = new_state_generator.generate_new_value(sa_system, i, T)
            sa_system.set_value(i, new_state)

    metadata = {
        "num_sweeps": num_sweeps,
        "state_updater": state_updater,
        "seed": seed,
        "T_min": T_min,
        "T_max": T_max,
    }

    return sa_system.get_state_dict(), sa_system.calc_energy(), metadata


def solve(
    interaction: Dict[Tuple[Any, ...], float],
    range_dict: Dict[Any, Tuple[int, int]],
    num_sweeps: int,
    num_samples: int = 1,
    T_min: Optional[float] = None,
    T_max: Optional[float] = None,
    state_updater: str = "METROPOLIS",
    seed: Optional[int] = None,
) -> Results:
    seed_list = [seed + i if seed is not None else None for i in range(num_samples)]
    result_list = []
    time_list = []
    for seed in seed_list:
        start = time.perf_counter()
        r = _solve(
            interaction, range_dict, num_sweeps, T_min, T_max, state_updater, seed
        )
        end = time.perf_counter()
        time_list.append(end - start)
        result_list.append(r)

    return Results(
        solution_list=[r[0] for r in result_list],
        energy_list=[r[1] for r in result_list],
        solving_time_list=time_list,
        meta_data_list=[r[2] for r in result_list],
    )
