import numpy as np
from typing import List, Tuple

class JC_state:
    def __init__(self, N: int, sign: bool):
        self._N = N
        self._sign = sign

    def lower_state(self):
        if self._N <= 0:
            return
        self._N -= 1
        self._sign = 0 if self._N == 0 else not self._sign

    def raise_state(self):
        self._N += 1
        self._sign = not self._sign

    def N_parity(self):
        return self._N % 2

    def generate_tuple(self):
        return (0, 0) if self._N == 0 else (self._N, int(self._sign))

    def get_JC_energy(self, g: float, w_q: float, w_r: float):
        # Convert GHz to MHz
        g *= 1e3
        w_q *= 1e3
        w_r *= 1e3

        n = self._N
        sign = self._sign
        delta = w_q - w_r
        return n * w_q + 0.5 * (delta + (-1) ** (sign + 1) * np.sqrt(delta**2 + 4 * n * g**2))

    def __eq__(self, other):
        return isinstance(other, JC_state) and (self._N == other._N) and (self._sign == other._sign)

def parse_state_string(s: str) -> Tuple[int, bool]:
    if len(s) < 2 or s[-1] not in ("m", "p"):
        raise ValueError(f"Invalid state format: {s}")
    return (int(s[:-1]), 0 if s[-1] == "m" else 1)

def parse_transition_string(s: str) -> Tuple[Tuple[int, bool], Tuple[int, bool]]:
    if "to" not in s:
        raise ValueError("Transition must be in format 'XmtoYn'")
    src, tgt = s.split("to")
    return parse_state_string(src), parse_state_string(tgt)

def generate_path_from_0(target_state_arr: Tuple[int, bool]) -> List[Tuple[int, bool]]:
    target = JC_state(*target_state_arr)
    start_N = 1
    start_sign = target._sign if target.N_parity() else not target._sign
    curr = JC_state(start_N, start_sign)
    path = [(0, 0), curr.generate_tuple()]
    while curr != target:
        curr.raise_state()
        path.append(curr.generate_tuple())
    return path

def generate_path_to_0(start_state: Tuple[int, bool]) -> List[Tuple[int, bool]]:
    start = JC_state(*start_state)
    zero = JC_state(0, 0)
    path = [start.generate_tuple()]
    while start != zero:
        start.lower_state()
        path.append(start.generate_tuple())
    return path

def generate_path(start_state_arr: Tuple[int, bool], target_state_arr: Tuple[int, bool]) -> List[Tuple[int, bool]]:
    start = JC_state(*start_state_arr)
    target = JC_state(*target_state_arr)
    if start == JC_state(0, 0):
        return generate_path_from_0(target_state_arr)
    elif (start.N_parity() == target.N_parity()) != (start._sign == target._sign) or (start._N == target._N):
        path = generate_path_to_0(start_state_arr)
        path += generate_path_from_0(target_state_arr)
        return path
    else:
        path = [start.generate_tuple()]
        curr = start
        while curr != target:
            curr.raise_state() if curr._N < target._N else curr.lower_state()
            path.append(curr.generate_tuple())
        return path

def get_transition_freqs(path: List[Tuple[int, bool]], g: float, w_q: float, w_r: float):
    freqs = []
    for i in range(len(path) - 1):
        s1 = JC_state(*path[i])
        s2 = JC_state(*path[i + 1])
        if s1 != s2:
            df = abs(s2.get_JC_energy(g, w_q, w_r) - s1.get_JC_energy(g, w_q, w_r)) / (2 * np.pi)
            freqs.append(df)
    return freqs
