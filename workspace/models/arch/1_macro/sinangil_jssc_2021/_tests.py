import sys
import os

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl
import scripts
# fmt: on


OUTPUT_SCALE_FACTOR = 0.25


def test_output_value_energy_scaling():
    """
    ### Data-Value-Dependent Energy

    This test replicates the results of Fig. 13(a) of the paper.

    The energy consumed by the macro is data-value-dependent. In this test, we
    will plot the average energy consumed by the macro as the average output
    value increases. We will vary the output value from 0 to 15. For each, we
    set all input and weight values to be the square root of the output value
    divided by a scaling factor. The scaling factor captures the scaling of
    outputs before the ADC and the summation of results from many rows.

    We see that the energy consumed by the macro increases as the average output
    value increases. This is because:

    - As the average input value increases, there is more switching on row wires
      to encode the larger input values, leading to higher energy.
    - Similarly, as the average weight value increases, there is greater
      discharging of summing capacitors on each column, again leading to higher
      energy.
    """
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                AVERAGE_INPUT_VALUE=OUTPUT_SCALE_FACTOR * (x**0.5),
                AVERAGE_WEIGHT_VALUE=OUTPUT_SCALE_FACTOR * (x**0.5),
                AVERAGE_OUTPUT_VALUE=x,
            ),
        )
        for x in range(16)
    )
    EXPECTED = {
        0: 3.35660091e-12,
        1: 4.562974203e-12,
        2: 5.033383915e-12,
        4: 5.443095599e-12,
        6: 5.837632777e-12,
        8: 6.285280728e-12,
        10: 6.657056146e-12,
        12: 7.04400607e-12,
        15: 7.795144158e-12,
    }
    for i, r in enumerate(results):
        r.add_compare_ref("energy", EXPECTED.get(i, None))

    return results


def test_voltage_scaling():
    """
    ### Voltage Scaling

    This test replicates the results of Fig. 13(b) of the paper.

    We show the area and energy of the macro at supply voltages ranging from
    0.65V to 1V, testing the worst-case (inputs, weights, and outputs are all
    maximum values) scenario. We see that the macro's energy consumption
    increases as the supply voltage increases.

    When developing the model, we found that there is a energy scales more
    aggressively with supply voltage in this result (Fig. 13(b) in the paper)
    than in the previous result (Table I). This difference could have been due
    to many factors in the fabricated chip measurement (temperature, input and
    output value distributions, clock frequency, etc.). We chose to set up the
    model to match Table I, leading to a discrepancy in this result.
    """

    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                VOLTAGE=x / 100,
                AVERAGE_INPUT_VALUE=OUTPUT_SCALE_FACTOR * (15**0.5),
                AVERAGE_WEIGHT_VALUE=OUTPUT_SCALE_FACTOR * (15**0.5),
                AVERAGE_OUTPUT_VALUE=15,
            ),
        )
        for x in range(65, 105, 5)
    )
    EXPECTED = [
        5.109375e-12,
        5.875e-12,
        6.796875e-12,
        7.8125e-12,
        9.03125e-12,
        10.25e-12,
        11.75e-12,
        13.125e-12,
    ]
    for r, e in zip(results, EXPECTED):
        r.add_compare_ref("energy", e)

    return results


def test_area_breakdown():
    """
    ### Area Breakdown

    This test replicates the results of Section IV paragraph 1 of the paper.

    We show the area of the macro and its subcomponents. We report the area of
    the CIM circuitry, the original macro, and the binary weighting capacitors.

    The CiM circuitry consumes the majority of the area, due to the high area of
    ADC and DAC circuitry.

    """
    results = utl.single_test(utl.quick_run(macro=MACRO_NAME))

    results.combine_per_component_area(["adc", "counter"], "CiM Circuitry")
    results.add_compare_ref_area("CiM Circuitry", [1596e-12])

    results.combine_per_component_area(
        [
            "row_drivers",
            "column_drivers",
            "weight_drivers",
            "cim_unit",
        ],
        "Original Macro",
    )
    results.add_compare_ref_area("Original Macro", [800e-12])

    results.combine_per_component_area(
        ["binary_weighting_capacitors"], "Binary Weighting Capacitors"
    )
    results.add_compare_ref_area("Binary Weighting Capacitors", [160e-12])

    return results


def test_tops():
    """
    ### Energy Efficiency and Throughput

    This test replicates the results of Table I in the paper. We show the area
    and energy efficiency and throughput of the macro at 0.8V and 1V supply
    voltages and at varying average output values. For output values, we use 0
    (minimum, best-case), 6 (average), and 15 (maximum, worst-case).

    We see that, as the supply voltage increases, the macro gains throughput at
    the cost of decreased energy efficiency. As the average output value
    increases, the energy efficiency also decreases due to the
    data-value-dependent energy consumption of row and column circuitry.

    There is significant variation only for the case with 1V supply voltage and
    average output value of 6. We attribute this to variation in measurements of
    the fabricated chip, as all reference results except for this case follow
    consistent scaling trends with supply voltage and average output value.
    Further tests on the fabricated chip may be required to find the precise
    cause of this difference.
    """
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                VOLTAGE=x,
                AVERAGE_INPUT_VALUE=OUTPUT_SCALE_FACTOR * (y**0.5),
                AVERAGE_WEIGHT_VALUE=OUTPUT_SCALE_FACTOR * (y**0.5),
                AVERAGE_OUTPUT_VALUE=y,
            ),
        )
        for x in [0.8, 1]
        for y in [15, 6, 0]
    )

    for r, tops, tops_per_w in zip(
        results,
        [372.4 * 1e-3] * 3 + [455.1 * 1e-3] * 3,
        [262.3, 351, 610.5, 189.3, 321, 435.5],
    ):
        r.add_compare_ref("tops", tops)
        r.add_compare_ref("tops_per_w", tops_per_w)

    return results


def test_exploration():
    """
    ### Exploration of CiM unit width versus number of weight bits

    This test explores the tradeoff between the width of the CiM unit (i.e.,
    number of weight bits that are stored and processed in one slice), the
    number of weight bits in the workload, and the compute density of the macro.

    In this test, we vary the CiM unit width while keeping the array size
    constant (*e.g.,* when we double the CiM unit width, it doubles the bits per
    weight slice but halves the number of columns). We then measure the
    throughput of the macro for different numbers of bits per weight. When
    changing the CiM unit width, the binary weighting capacitors will also be
    scaled to sum results from the bits within a weight slice.

    We see that CiM units with more weight bits can increase compute density for
    a given number of weight bits because they store more bits in each slice,
    require fewer columns to store slices, and require less circuitry (mostly
    ADCs) to read outputs. However, CiM units become underutilized when there
    are fewer bits per weight than they store.

    Wider CiM units also lead to a larger-area chip due to the larger binary
    weighting capacitors required to sum results from the bits within a weight
    slice. The size of the binary weighting capacitors increases exponentially
    with the number of bits per weight. For this reason, the eight-wide CiM unit
    had high area from the binary weighting capacitors, and it never had the
    highest compute density.
    """
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                CIM_UNIT_WIDTH_CELLS=x,
                N_COLUMNS=64 // x,
                WEIGHT_BITS=y,
            ),
        )
        for x in [1, 2, 4, 8]
        for y in range(1, 9)
    )

    return results


if __name__ == "__main__":
    test_tops(),
    test_output_value_energy_scaling(),
    test_voltage_scaling(),
    test_area_breakdown(),
    test_exploration(),
