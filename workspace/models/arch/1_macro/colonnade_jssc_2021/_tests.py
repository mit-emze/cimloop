import sys
import os

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl
import scripts
# fmt: on


def test_energy_breakdown():
    """
    ### Energy Breakdown

    This test replicates the results presented in Fig. 17 of the paper. Energy
    is measured using 8b weights.

    We show the energy breakdown fo the macro. The energy is broken down into
    the following components:

    - Bitcell Array: The energy consumed by the SRAM cells and digital CiM
      circuitry (1b multiply, adders, configuration) in the array.
    - Drivers: The energy consumed by the array drivers. Note that we model a
      transposed version of the macro, so our drivers are modeled as wordline
      drivers while the Colonnade paper uses bitline drivers.
    - Ctrl+Reg: The energy consumed by the control logic and registers.
      Registers include those inside the array that accomplish in-array
      pipelining.
    - BL Decoder: Note that in Colonnade this is actually a WL decoder, but
      we're transposed in this model.

    We can see that the bitcell array consumes the majority of the macro's
    energy. This is because the digital CiM macro requires a full adder and
    configuration circuits to be connected to every SRAM bitcell, resulting in
    high area and energy overhead for the SRAM array.

    """
    results = utl.single_test(
        utl.quick_run(
            macro=MACRO_NAME,
            variables=dict(
                INPUT_BITS=1,
                WEIGHT_BITS=8,
                OUTPUT_BITS=16,
                FORCE_100MHZ=True,
            ),
        )
    )
    results.combine_per_component_energy(["cim_unit", "cim_logic"], "Bitcell array")
    results.add_compare_ref_energy("Bitcell array", 0.602 * 0.1443 * 128 * 8 * 1e-12)
    results.combine_per_component_energy(
        ["row_drivers", "digital_logic_input_ports"], "Drivers"
    )
    results.add_compare_ref_energy("Drivers", 0.288 * 0.1443 * 128 * 8 * 1e-12)
    results.combine_per_component_energy(["register"], "Ctrl+Reg")
    results.add_compare_ref_energy("Ctrl+Reg", 0.075 * 0.1443 * 128 * 8 * 1e-12)
    results.combine_per_component_energy(["column_drivers"], "BL Decoder")
    results.add_compare_ref_energy("BL Decoder", 0.035 * 0.1443 * 128 * 8 * 1e-12)
    results.clear_zero_energies()
    return results


def test_area_breakdown():
    """
    ### Area Breakdown

    This test replicates the results presented in Fig. 17 of the paper.

    We show the area breakdown fo the macro. The area is broken down into the
    following components:

    - Bitcell Array: The energy consumed by the SRAM cells and digital adders in
      the array.
    - Drivers: The energy consumed by the array drivers. Note that we model a
      transposed version of the macro, so our drivers are modeled as wordline
      drivers while the Colonnade paper uses bitline drivers.
    - Ctrl+Reg: The energy consumed by the control logic and registers.
      Registers include those inside the array that accomplish in-array
      pipelining.
    - BL Decoder: Note that in Colonnade this is actually a WL decoder, but
      we're transposed in this model.

    We can see that the bitcell array consumes the majority of the macro's area.
    This is because the digital CiM macro requires a full adder and
    configuration circuits to be connected to every SRAM bitcell, resulting in
    high area and energy overhead for the SRAM array.
    """
    results = utl.single_test(
        utl.quick_run(
            macro=MACRO_NAME,
            variables=dict(
                INPUT_BITS=1,
                WEIGHT_BITS=8,
                OUTPUT_BITS=16,
                FORCE_100MHZ=True,
            ),
        )
    )
    results.combine_per_component_area(["cim_unit", "cim_logic"], "Bitcell array")
    results.add_compare_ref_area("Bitcell array", 0.839 * 227200e-12)
    results.combine_per_component_area(
        ["row_drivers", "digital_logic_input_ports"], "Drivers"
    )
    results.add_compare_ref_area("Drivers", 0.018 * 227200e-12)
    results.combine_per_component_area(["register"], "Ctrl+Reg")
    results.add_compare_ref_area("Ctrl+Reg", 0.125 * 227200e-12)
    results.combine_per_component_area(["column_drivers"], "BL Decoder")
    results.add_compare_ref_area("BL Decoder", 0.018 * 227200e-12)
    results.clear_zero_areas()
    return results


def test_energy_input_weight_bits_scaling():
    """
    # Energy vs. Number of Input and Weight Bits

    This test replicates the results presented in Fig. 16a of the paper. Like
    with the energy versus weight bits test, we'll test three voltages and
    measure energy 1-16b weights. This time, we vary the number of input bits as
    well, using the same number of input bits as weight bits.

    We can see that, in addition to the energy increasing with the number of
    weight bits, the energy also increases linearly with the number of input
    bits because an N-bit input is processed in N cycles and N activations of
    the array.  We also fix the clock frequency at 100MHz, as is done in the
    paper (for measurement purposes; the macro can support higher frequencies).

    Modeled and reference results vary for the same two reasons as the energy
    versus weight bits test, and could be made closer by using a more complex
    data-value-dependent calculation for the energy of the adders in the array,
    and by using a more complex voltage-energy scaling function.
    """
    EXPECTED_FJ_PER_MAC = {
        # fmt: off
        #    0.6V        0.7V          0.8V
        1:  [17.1320369,    42.5577975,   68.10051312],
        2:  [39.750389,     101.7850194,  161.6449048],
        3:  [72.35941721,   183.8842178,  292.0267348],
        4:  [104.1269428,   260.6311646,  423.4320763],
        5:  [143.1768052,   369.409647,   591.1251988],
        6:  [198.3696842,   507.9462783,  788.5272112],
        7:  [215.6250592,   543.8204606,  844.217685],
        8:  [292.0267348,   725.4253956,  1169.653111],
        9:  [310.2896494,   788.5272112,  1252.260958],
        10: [420.2334993,   1067.923309,  1695.970219],
        11: [439.7938751,   1109.188833,  1748.197762],
        12: [613.9668116,   1536.767343,  2422.106272],
        13: [632.8739703,   1584.09222,   2554.140514],
        14: [642.5447846,   1645.302977,  2632.795482],
        15: [960.366235,    2459.11797,   3905.328036],
        16: [989.9407926,   2515.698651,  3995.183881],
        # fmt: on
    }

    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                INPUT_BITS=i,
                WEIGHT_BITS=i,
                OUTPUT_BITS=16,
                VOLTAGE=v,
                FORCE_100MHZ=True,
            ),
        )
        for i in EXPECTED_FJ_PER_MAC
        for v in [0.6, 0.7, 0.8]
    )

    expected_flattened = [e2 for e in EXPECTED_FJ_PER_MAC.values() for e2 in e]

    for r, e in zip(results, expected_flattened):
        r.add_compare_ref("energy", e * 1e-15 * r.computes)  # -> fJ
    return results


def test_energy_weight_bits_scaling():
    """
    ### Energy vs. Number of Weight Bits

    This test replicates the results presented in Fig. 16b of the paper. The
    macro supports between 1 and 16 weight bits. For three different voltages,
    the energy per MAC of the macro is measured for varying numbers of weight
    bits. We fix the number of input bits to 1. We also fix the clock frequency
    at 100MHz, as is done in the paper (for measurement purposes; the macro can
    support higher frequencies).

    We see that the number of weight bits scales energy because of two factors.

    - First, as the number of weight bits increases, the number of adders
      required to compute MAC results increases, leading to higher energy
      consumption. The number of full adders required to compute each MAC is
      equal to the number of weight bits plus 7. This is because the weights are
      stored in the array and the accumulation results of up to 128 MACs are
      propagated through adders. The array therefore requires one adder for
      each weight bit plus an additional 7 adders per weight to store the
      full precision of the accumulation results.
    - Second, as the number of weight bits increases, the number of parallel
      MACs that the array can perform decreases. This will lead to a decrease in
      the number of parallel MACs, and thus an increase in energy per MAC. This
      effect leads to steps in the energy each time the number of parallel MACs
      decreases. For example, there is a step when going from 14b (21 columns
      per weight, 6 parallel MACs per row) to 15b weights (22 columns per
      weight, 5 parallel MACs per row).

    We also see that the energy per MAC increases with voltage.

    There are two significant differences between published and modeled results:

    - In the published results, the energy per MAC does not increase
      monotonically with the number of weight bits. Energy per MAC may decrease
      from N to N+1 bit weights if the number of parallel MACs is the same for
      both N and N+1 bit weights (i.e., the array utilization is the same for
      both). We attribute this to a change in data-value-dependent switching
      activity of adders in the array, and this effect could be captured with a
      more complex data-value-dependent calculation.
    - In the published results, the energy per MAC decreases rapidly as supply
      voltage decreases (~1.6x from 0.8V to 0.7V and ~4x from 0.8V to 0.6V).
      This is likely due to some technology-specific effects. We could model it
      by using a piecewise function for VOLTAGE_ENERGY_SCALE and passing it to
      all subcomponent models, but we instead chose to use simple V^2 scaling.

    """

    EXPECTED_FJ_PER_MAC = {
        # fmt: off
        #    0.6V        0.7V          0.8V
        1:  [17.07820589,   42.83967039,    68.31697711],
        2:  [20.27486225,   49.48109179,    77.83242912],
        3:  [24.0698594,    59.9648546, 	96.94840357],
        4:  [26.49734159,   66.92475081,    105.2708593],
        5:  [29.37052503,   73.6742141, 	115.0949303],
        6:  [33.23249487,   83.93585286,    131.125787],
        7:  [30.81598573,   77.30009326,    121.5909692],
        8:  [36.83598758,   91.76892401,    144.3500315],
        9:  [34.39268536,   85.09591799,    133.8535776],
        10: [42.83967039,   106.725807, 	166.7286341],
        11: [39.72457364,   99.64675226,    158.9079843],
        12: [51.56119566,   126.7024855,    204.8466723],
        13: [49.82184952,   121.5909692,    189.9512015],
        14: [47.16012347,   115.0949303,    181.041186],
        15: [63.7855784,    157.8210593,    255.1576938],
        16: [59.9648546,    151.4542256,    243.1891829],
        # fmt: on
    }
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                INPUT_BITS=1,
                WEIGHT_BITS=i,
                OUTPUT_BITS=16,
                VOLTAGE=v,
                FORCE_100MHZ=True,
            ),
        )
        for i in EXPECTED_FJ_PER_MAC
        for v in [0.6, 0.7, 0.8]
    )

    expected_flattened = [e2 for e in EXPECTED_FJ_PER_MAC.values() for e2 in e]
    for r, e in zip(results, expected_flattened):
        r.add_compare_ref("energy", e * 1e-15 * r.computes)  # -> fJ

    return results


def test_throughput_scaling():
    """
    ### Throughput vs. Number of Registers per Column

    This test replicates the results presented in Fig. 15a of the paper. The
    number of registers in the macro is varied from 1 to 16. A register in the
    macro is a flip flop, one for each column in the array, that breaks up the
    critical path length into a smaller number of rows to allow for higher clock
    frequencies.

    In addition to varying the number of registers, the number of input and
    weight bits are varied from 1 to 16.

    Throughput increases with the number of registers as the critical path
    latency reduces. Effects begin to saturate with many registers as the
    critical path begins to become dominated by the full adders communicating
    between columns rather than between rows (i.e., carry signals propagating
    between adders in the same row, rather than sum signals propagating between
    adders in the same column).

    We also find that throughput decreases approximately quadratically as we
    increase the number of input and weight bits because the number of clock
    cycles increases linearly with the number of input bits and the number of
    computations computed by the array decreases approximately linearly with the
    number of weight bits.

    """
    EXPECTED_THROUGHPUT = {
        # fmt: off
        #    1 reg      2 regs      4       8       16
        1:  [75.07,     138.51,     243.93,	391.42,	563.50],
        2:  [32.50,     60.42,	    104.77,	168.13,	238.32],
        3:  [18.31,     34.04,	    59.03,	92.55,	127.18],
        4:  [12.52,     23.10,	    39.45,	60.42,	83.03],
        5:  [9.11,      16.43,	    28.05,	42.96,	58.12],
        6:  [6.73,      12.33,	    20.41,	31.26,	41.65],
        7:  [5.81,      10.24,	    17.21,	25.75,	34.84],
        8:  [4.43,      7.92,	    13.32,	19.48,	25.75],
        9:  [3.91,      6.95,	    11.41,	16.81,	22.23],
        10: [3.08,      5.38,	    8.77,	12.72,	16.68],
        11: [2.76,      4.79,	    7.99,	11.50,	14.51],
        12: [2.14,      3.74,	    6.09,	8.56,	11.06],
        13: [1.99,      3.48,	    5.46,	7.62,	9.77],
        14: [1.82,      3.13,	    4.94,	6.95,	8.90],
        15: [1.43,      2.42,	    3.79,	5.25,	6.63],
        16: [1.34,      2.26,	    3.51,	4.79,	5.99],
        # fmt: on
    }

    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                INPUT_BITS=i,
                WEIGHT_BITS=i,
                N_REGS_PER_COL=[1, 2, 4, 8, 16][j],
                N_ROWS_PER_REG=128 // [1, 2, 4, 8, 16][j],
                FORCE_100MHZ=False,
                OUTPUT_BITS=16,
            ),
        )
        for i in range(1, 17)
        for j in range(5)
    )

    expected_flattened = [e2 for e in EXPECTED_THROUGHPUT.values() for e2 in e]
    for r, e in zip(results, expected_flattened):
        r.add_compare_ref("tops", e / 1000)  # GOPS -> TOPS

    return results


def test_latency_scaling():
    """
    ### Latency vs. Number of Registers per Column

    This test replicates the results presented in Fig. 16b of the paper. Like in
    the previous test, the number of registers in the macro is varied from 1 to
    16. The latency of the macro is measured for varying numbers of registers.

    Note that in other tests, the macro is bit pipelined, meaning that when one
    pipeline stage (a stage being a row group in the array) finishes processing
    a bit, it is immediately filled with the next bit even if all pipeline stages
    are not finished processing the current bit. In this test, the macro is NOT
    bit pipelined, meaning that all pipeline stages need to be flushed before
    processing the next bit. This significantly increases the latency of the
    macro.

    We find first that latency increases approximately linearly with the number
    of input bits because each additional input bit requires an additional clock
    cycle to process. We find that, at first, latency decreases with additional
    registers because the critical path length of the array is reduced. However,
    at a certain point (2-4 registers depending on the number of input bits),
    latency begins to increase because flushing the pipeline requires additional
    clock cycles.
    """
    EXPECTED_LATENCY = {
        # fmt: off
        # 1 reg col 2       4       8       16
        1:  [0.12,  0.09,   0.09,   0.10,   0.13],
        2:  [0.22,  0.18,   0.17,   0.20,   0.26],
        3:  [0.34,  0.27,   0.26,   0.30,   0.41],
        4:  [0.46,  0.37,   0.37,   0.42,   0.58],
        5:  [0.57,  0.47,   0.46,   0.54,   0.75],
        6:  [0.68,  0.57,   0.56,   0.67,   0.95],
        7:  [0.80,  0.67,   0.67,   0.82,   1.15],
        8:  [0.94,  0.78,   0.79,   0.96,   1.37],
        9:  [1.05,  0.88,   0.91,   1.11,   1.59],
        10: [1.17,  1.00,   1.03,   1.26,   1.85],
        11: [1.30,  1.11,   1.15,   1.44,   2.09],
        12: [1.44,  1.22,   1.27,   1.61,   2.37],
        13: [1.55,  1.34,   1.41,   1.78,   2.66],
        14: [1.69,  1.46,   1.54,   1.96,   2.95],
        15: [1.82,  1.58,   1.69,   2.16,   3.27],
        16: [1.95,	1.71,	1.82,	2.36,	3.60],
        # fmt: on
    }

    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                INPUT_BITS=i,
                WEIGHT_BITS=i,
                N_REGS_PER_COL=[1, 2, 4, 8, 16][j],
                N_ROWS_PER_REG=128 // [1, 2, 4, 8, 16][j],
                FORCE_100MHZ=False,
                OUTPUT_BITS=16,
                BIT_PIPELINED=False,
            ),
        )
        for i in range(1, 17)
        for j in range(5)
    )

    expected_flattened = [e2 for e in EXPECTED_LATENCY.values() for e2 in e]
    for r, e in zip(results, expected_flattened):
        r.add_compare_ref("latency", e / 1e6)  # ->us
    return results


if __name__ == "__main__":
    test_energy_breakdown()
    test_area_breakdown()
    test_energy_input_weight_bits_scaling()
    test_energy_weight_bits_scaling()
    test_throughput_scaling()
    test_latency_scaling()
