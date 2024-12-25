from simulated_annealing.integer.sa_system import SASystem
import math


class StateGeneratorMetropolis:
    def __init__(self):
        pass

    def generate_new_value(self, sa_system: SASystem, index: int, T: float) -> int:
        candidate_value = sa_system.var_list[index].get_candidate_value()
        dE = sa_system.get_dE(index, candidate_value)
        rand = sa_system.random.random()
        if dE <= 0.0 or rand < math.exp(-dE / T):
            return candidate_value
        else:
            return sa_system.var_list[index].value


class StateGeneratorHeatBath:
    def __init__(self, max_num_state: int):
        self.plob_list = [0 for _ in range(max_num_state)]
        self.dE_list = [0 for _ in range(max_num_state)]

    def generate_new_value(self, sa_system: SASystem, index: int, T: float) -> int:
        var = sa_system.var_list[index]
        min_dE = float("inf")
        for state in range(var.num_states):
            value = var.get_value_from_state(state)
            self.dE_list[state] = sa_system.get_dE(index, value)
            min_dE = min(min_dE, self.dE_list[state])
        z = 0.0
        for state in range(var.num_states):
            dE = self.dE_list[state]
            self.plob_list[state] = math.exp(-(dE - min_dE) / T)
            z += self.plob_list[state]
        z = 1.0 / z
        prob_sum = 0.0
        rand = sa_system.random.random()
        for state in range(var.num_states):
            prob_sum += self.plob_list[state] * z
            if rand < prob_sum:
                return var.get_value_from_state(state)
        return var.get_value_from_state(var.num_states - 1)


class StateGeneratorSuwaToda:
    def __init__(self, max_num_state: int):
        self.weight_list = [0 for _ in range(max_num_state)]
        self.sum_weight_list = [0 for _ in range(max_num_state + 1)]
        self.dE_list = [0 for _ in range(max_num_state)]

    def generate_new_value(self, sa_system: SASystem, index: int, T: float) -> int:
        var = sa_system.var_list[index]
        max_weight_state, min_dE = sa_system.get_max_weight_state(index)

        for state in range(var.num_states):
            value = var.get_value_from_state(state)
            self.dE_list[state] = sa_system.get_dE(index, value) - min_dE

        self.weight_list[0] = math.exp(-self.dE_list[max_weight_state] / T)
        self.sum_weight_list[1] = self.weight_list[0]

        for state in range(1, var.num_states):
            if state == max_weight_state:
                self.weight_list[state] = math.exp(-self.dE_list[0] / T)
            else:
                self.weight_list[state] = math.exp(-self.dE_list[state] / T)
            self.sum_weight_list[state + 1] = (
                self.sum_weight_list[state] + self.weight_list[state]
            )

        self.sum_weight_list[0] = self.sum_weight_list[var.num_states]
        now_state = (
            max_weight_state
            if var.state == 0
            else (0 if var.state == max_weight_state else var.state)
        )
        prob_sum = 0.0
        rand = sa_system.random.random()
        for j in range(var.num_states):
            d_ij = (
                self.sum_weight_list[now_state + 1]
                - self.sum_weight_list[j]
                + self.sum_weight_list[1]
            )
            prob_sum += max(
                0.0,
                min(d_ij, 1.0 + self.weight_list[j] - d_ij, 1.0, self.weight_list[j]),
            )
            if rand < prob_sum:
                new_state = (
                    0 if j == max_weight_state else (max_weight_state if j == 0 else j)
                )
                return var.get_value_from_state(new_state)
        return var.get_value_from_state(var.num_states - 1)
