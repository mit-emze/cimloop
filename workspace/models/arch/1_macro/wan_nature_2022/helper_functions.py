from math import ceil, floor
from typing import Tuple


def bits2phase_1phase(n_bits: int,  max_bits_per_phase: int = 0) -> Tuple[int]:
    return (n_bits,)


def bits2phase_2phase(n_bits: int, max_bits_per_phase: int) -> Tuple[int, int]:
    if n_bits <= max_bits_per_phase:
        return (n_bits, 0)
    return (ceil(n_bits / 2), floor(n_bits / 2))


def bits2integration_steps(n_bits: int) -> int:
    return 2 ** n_bits - 1


def phase_out_bits(out_bits: int, phases: Tuple[int], phasenum: int) -> int:
    return max(out_bits - sum(phases[:phasenum]), 0)


def sum_phase_out_bits(total_out_bits: int, phases: Tuple[int]) -> int:
    s = 0
    for i, p in enumerate(phases):
        if p == 0:
            continue
        s += phase_out_bits(total_out_bits, phases, i)
    return s


def sum_map(f: callable, *args):
    args = [a for a in args]
    max_len = max(len(a) if isinstance(a, (list, tuple)) else a for a in args)
    for i, a in enumerate(args):
        if not isinstance(a, (list, tuple)) or len(a) != max_len:
            args[i] = [a] * max_len
    return sum(map(f, *args))
