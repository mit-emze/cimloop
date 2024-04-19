# Sign magnitude
# 1. Scale X to [-1, 1]: x = NORM_1_TO_NEG1(x, INPUTS_VALUE_DISTRIBUTION)
# 2. Convert to signed: x = abs(x) * (2 ** (INPUT_BITS - 1) - 1)
# 2. x = round(x * (2 ** INPUT_BITS - 1))

from math import log2
from typing import List, NamedTuple, Union


class ProbableBits(NamedTuple):
    bits: list
    probability: float


def assert_hist_pow2_minus1(hist):
    x = 1
    while x <= len(hist):
        x *= 2
    assert x - 1 == len(
        hist
    ), f"Histogram length {len(hist)} is not a power of 2 minus 1."


def norm_encoded_hist(encoded_hist: List[ProbableBits]):
    sum_probs = sum([e.probability for e in encoded_hist])
    return [ProbableBits(e.bits, e.probability / sum_probs) for e in encoded_hist]


def get_num_bits(hist):
    n_bits = 0
    while 2**n_bits < len(hist) + 1:
        n_bits += 1
    assert (
        2**n_bits == len(hist) + 1
    ), f"Number of histogram bins + 1 must be a power of 2, got {len(hist)}."
    return n_bits


def is_hist_signed(hist):
    return sum(hist[: len(hist) // 2]) != 0


def hist_to_magnitude(hist):
    assert_hist_pow2_minus1(hist)
    new_hist = [0] * (len(hist) // 2)
    hist_center = len(hist) // 2
    for i in range(len(new_hist)):
        new_hist[i] = hist[hist_center + i] + hist[hist_center - i]
    assert_hist_pow2_minus1(new_hist)
    return new_hist


def magnitude_encode_hist(weights) -> List[ProbableBits]:
    nbits = get_num_bits(weights)
    encoded = []
    halfwidth = len(weights) / 2
    for i, w in enumerate(weights):
        normed = norm(i, len(weights), -halfwidth + 0.5, halfwidth + 0.5)
        encoded.append(ProbableBits(to_bits_unsigned(abs(normed), nbits)[1:], w))
    return norm_encoded_hist(encoded)


def offset_encode_hist(weights):
    nbits = get_num_bits(weights)
    encoded = []
    for i, w in enumerate(weights):
        normed = norm(i, len(weights), 0, len(weights))
        encoded.append(ProbableBits(to_bits_unsigned(normed, nbits), w))
    return norm_encoded_hist(encoded)


def offset_encode_if_signed_hist(weights):
    if is_hist_signed(weights):
        return offset_encode_hist(weights)
    return magnitude_encode_hist(weights)


def two_sided_encode_hist(weights):
    m = magnitude_encode_hist(weights)
    m2 = []
    for e in m:
        m2.append(ProbableBits(e.bits, e.probability / 2))
        m2.append(ProbableBits([0] * len(e.bits), e.probability / 2))
    return m2


def two_sided_encode_if_signed_hist(weights):
    if is_hist_signed(weights):
        return two_sided_encode_hist(weights)
    return magnitude_encode_hist(weights)


def xnor_encode_hist(weights):
    nbits = get_num_bits(weights)
    encoded = []
    halfwidth = len(weights) / 2
    for i, w in enumerate(weights):
        normed = norm(i, len(weights), -halfwidth + 0.5, halfwidth + 0.5)
        bits = []
        for j in list(range(nbits - 1, -1, -1)) + [-1, -1]:
            bits.append(int(normed > 0))
            normed -= 2**j * (2 * bits[-1] - 1)
        assert normed == 0, f"normed={normed} is not 0"
        encoded.append(ProbableBits(bits, w))
    return norm_encoded_hist(encoded)


def zero_gated_xnor_encode_hist(weights):
    encoded = xnor_encode_hist(weights)
    zero_idx = len(encoded) // 2
    encoded[zero_idx] = ProbableBits(
        [0] * len(encoded[zero_idx].bits), encoded[zero_idx].probability
    )
    return encoded


def to_bits_unsigned(x, nbits):
    x = round(x)
    assert 0 <= x < 2**nbits, f"x={x} is not in range [0, 2^{nbits})"
    return [int(i) for i in bin(x)[2 : nbits + 2].zfill(nbits)]


def norm(x, nbins, rmin, rmax):
    return x / nbins * (rmax - rmin) + rmin


def encoded_hist_to_avg_slice(
    encoded_hist: List[ProbableBits],
    total_bits: int,
    bits_per_slice: Union[list, int],
    partial_slices_use_full_range: bool = False,
    return_per_slice: bool = False,
):
    if isinstance(bits_per_slice, int):
        bits_per_slice = [bits_per_slice] * (total_bits // bits_per_slice)
        if sum(bits_per_slice) != total_bits:
            bits_per_slice.append(total_bits - sum(bits_per_slice))

    assert total_bits == sum(bits_per_slice), (
        f"Sum of bits per slice {sum(bits_per_slice)} != total_bits " f"{total_bits}"
    )

    bit2slice = []
    max_val = max(2 ** max(bits_per_slice) - 1, 1)
    for i, b in enumerate(bits_per_slice):
        m = max(2**b - 1, 1) if partial_slices_use_full_range else max_val
        bit2slice += [(i, max((2 ** (b - j - 1)), 1) / m) for j in range(b)]

    avg_slice_values = [0] * len(bits_per_slice)
    for e in encoded_hist:
        for i in range(total_bits):
            slice_idx, scale = bit2slice[i]
            bit_value = e.bits[i] if i < len(e.bits) else 0.5
            avg_slice_values[slice_idx] += bit_value * e.probability * scale

    if return_per_slice:
        return avg_slice_values

    return sum(avg_slice_values) / len(avg_slice_values)


if __name__ == "__main__":
    input_dist = [16 - abs(16 - i) for i in range(31)]
    print(f"input_dist: {input_dist}")
    for e in xnor_encode_hist(input_dist):
        print(e)
