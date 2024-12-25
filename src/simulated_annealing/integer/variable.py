import random


class Variable:
    def __init__(self, lower_bound: int, upper_bound: int, seed: int):
        if lower_bound > upper_bound:
            raise ValueError("Lower bound must be less than or equal to upper bound")
        self.random = random.Random(seed)
        self.lower_bound = lower_bound
        self.num_states = upper_bound - lower_bound + 1
        self.state = self.random.randrange(0, self.num_states)
        self.value = self.lower_bound + self.state

    def get_candidate_value(self) -> int:
        candidate_state = self.random.randrange(0, self.num_states - 1)
        if candidate_state >= self.state:
            candidate_state += 1
        return self.lower_bound + candidate_state

    def get_value_from_state(self, state: int) -> int:
        return self.lower_bound + state

    def set_value(self, new_value: int) -> None:
        self.state = new_value - self.lower_bound
        self.value = new_value
