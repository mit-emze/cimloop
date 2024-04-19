import math
from typing import List


def rescale_sum_to_1(array: List[float], do_not_change_index: int = -1) -> List[float]:
    """Rescales all list elements such that the sum is 1."""
    sum_array = sum([a for i, a in enumerate(array) if i != do_not_change_index])
    target_sum = 1 - array[do_not_change_index] if do_not_change_index >= 0 else 1
    scaleby = target_sum / sum_array
    print([a * scaleby if i != do_not_change_index else a for i, a in enumerate(array)])
    return [a * scaleby if i != do_not_change_index else a for i, a in enumerate(array)]


def set_element_rescale_sum_to_1(array: List[float], index: int, value: float):
    """Sets an element of a list, then rescales all list elements such that the sum is 1."""
    array[index] = value
    return rescale_sum_to_1(array, index)


def value2bits(value: int, resolution: int) -> List[int]:
    """Converts a value to a list of bits."""
    return [int(i) for i in bin(value)[2:].zfill(resolution)]


def bit_distribution_2_hist(
    bit_distribution: List[float], zero_prob: float = None
) -> List[float]:
    """Converts a bit distribution to a value distribution."""
    hist = [1] * 2 ** len(bit_distribution)
    for value in range(2 ** len(bit_distribution)):
        bits = value2bits(value, len(bit_distribution))
        for i, prob in enumerate(bit_distribution):
            hist[value] *= prob if bits[i] else 1 - prob

    if zero_prob is not None:
        set_element_rescale_sum_to_1(hist, 0, zero_prob)
    return rescale_sum_to_1(hist)


def hist_2_bit_distribution(hist: List[float]) -> List[float]:
    """Converts a value distribution to a bit distribution."""
    sum_hist = sum(hist)
    hist = [i / sum_hist for i in hist]

    bit_distribution = [0] * math.ceil(math.log(len(hist), 2))
    for value in range(len(hist)):
        for i, bit in enumerate(value2bits(value, len(bit_distribution))):
            bit_distribution[i] += hist[value] * bit

    return bit_distribution
