from typing import Optional, Union, List, Tuple, Dict, Any

from simulated_annealing.integer.variable import Variable
from collections import defaultdict
import random
import math


class SASystem:
    def __init__(
        self,
        interaction: Dict[Tuple[Any, ...], float],
        range_dict: Dict[Any, Tuple[int, int]],
        seed: Optional[int] = None,
    ) -> None:
        reduced_interaction = defaultdict(float)
        for key, value in interaction.items():
            if not isinstance(key, tuple):
                raise ValueError("Invalid interaction. key must be tuple")
            reduced_interaction[tuple(sorted(key))] += value

        self.index_list = sorted(set(x for key in reduced_interaction for x in key))
        self.index_map = {x: i for i, x in enumerate(self.index_list)}
        self.size = len(self.index_list)
        self.J_list, self.h_list, self.self_J_list = self.divide_interactions(
            reduced_interaction, self.index_map
        )
        self.random = random.Random(seed)
        self.var_list = self.generate_var_list(
            self.index_list, range_dict, self.random.randint(0, 2**32 - 1)
        )
        self.dE_list = self.calc_dE_list(self.J_list, self.h_list, self.var_list)

    def calc_dE_list(
        self,
        J_list: List[List[Tuple[int, float]]],
        h_list: List[float],
        var_list: List[Variable],
    ) -> List[float]:
        dE_list = [0] * len(var_list)

        for i in range(len(var_list)):
            dE = h_list[i]
            for v in J_list[i]:
                dE += v[1] * var_list[v[0]].value
            dE_list[i] = dE

        return dE_list

    def divide_interactions(
        self, interaction: Dict[Tuple[Any, ...], float], index_map: Dict[int, int]
    ) -> Tuple[List[List[Tuple[int, float]]], List[float], List[float]]:
        J_list = [[] for _ in index_map]
        h_list = [0] * len(index_map)
        self_J_list = [0] * len(index_map)

        for key, value in interaction.items():
            if len(key) == 1:
                (i,) = key
                i = index_map[i]
                h_list[i] = value
            elif len(key) == 2:
                i, j = key
                i = index_map[i]
                j = index_map[j]
                if i == j:
                    self_J_list[i] = value
                else:
                    J_list[i].append((j, value))
                    J_list[j].append((i, value))
            else:
                raise ValueError("Invalid interaction")

        for i in range(len(J_list)):
            J_list[i].sort()

        return J_list, h_list, self_J_list

    def generate_var_list(
        self, index_list: List[int], range_dict: Dict[int, Tuple[int, int]], seed: int
    ) -> List[Variable]:
        random_number_engine = random.Random(seed)
        var_list = []
        for ind in index_list:
            lower, upper = range_dict.get(ind, (0, 1))
            variable_seed = random_number_engine.randint(0, 2**32 - 1)
            var_list.append(Variable(lower, upper, variable_seed))

        return var_list

    def get_dE(self, index: int, candidate_value: int) -> float:
        a = candidate_value - self.var_list[index].value
        return a * (
            self.dE_list[index]
            + self.self_J_list[index] * (2 * self.var_list[index].value + a)
        )

    def set_value(self, index: int, new_value: int) -> None:
        if new_value == self.var_list[index].value:
            return
        for i, J in self.J_list[index]:
            self.dE_list[i] += J * (new_value - self.var_list[index].value)
        self.var_list[index].set_value(new_value)

    def calc_energy(
        self, var_list: Union[List[int], List[Variable], None] = None
    ) -> float:
        if var_list is None:
            var_list = self.var_list

        if isinstance(var_list[0], Variable):
            var_list = [v.value for v in var_list]

        energy = 0
        for i in range(len(var_list)):
            energy += self.h_list[i] * var_list[i]
            energy += self.self_J_list[i] * var_list[i] * var_list[i]
            for j, J in self.J_list[i]:
                energy += 0.5 * J * var_list[i] * var_list[j]
        return energy

    def get_max_weight_state(self, index: int) -> Tuple[int, float]:
        min_index = -1
        min_val = float("inf")
        for state in range(self.var_list[index].num_states):
            val = self.get_dE(index, state + self.var_list[index].lower_bound)
            if val < min_val:
                min_val = val
                min_index = state

        return min_index, min_val

    def get_state_dict(self) -> Dict[Any, int]:
        reverse_map = {v: k for k, v in self.index_map.items()}
        return {reverse_map[i]: self.var_list[i].value for i in range(self.size)}

    def get_min_max_temperature(self) -> Tuple[float, float]:
        min_dE = float("inf")
        max_dE = -float("inf")
        for i in range(self.size):
            for state in range(self.var_list[i].num_states):
                dE = abs(self.get_dE(i, self.var_list[i].get_value_from_state(state)))
                if dE > 1e-07 and dE < min_dE:
                    min_dE = dE
                if dE > 1e-07 and dE > max_dE:
                    max_dE = dE

        T_max = max_dE / math.log(4)
        T_min = min_dE / math.log(100)
        return T_max, T_min
