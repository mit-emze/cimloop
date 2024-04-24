import sys
import os

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl
import scripts
# fmt: on


# 256x256 array requires 2 devices per signed weight --> 256 * 128 MACs
# 2 ops/MAC -> 256 * 128 * 2 ops
OPS_PER_ARRAY = 256 * 128 * 2
WEIGHT_BITS = 4


def test_tops():
    """
    ### Energy Efficiency and Throughput

    This test replicates the results of Fig. 12 (c) and (d) of the paper.

    We show the area and energy efficiency and throughput of the macro at
    varying numbers of input and output bits. We also compare two-phase and
    one-phase operation.

    Note that this macro uses a signed DAC, which is underutilized for 1b
    inputs. For this reason, 1b and 2b inputs required the same number
    of activations for most circuits, and therefore the energy and throughput
    numbers are similar for 1b and 2b inputs.

    We see that increasing the number of input and output bits decreases the
    energy efficiency and throughput of the macro. This is for two reasons.
    First, processing each additional input bit requires an extra cycle and
    extra activation of most macro components. Second, each additional output
    bit requires additional energy to be consumed by this macro's
    variable-precision ADC.

    We also see that two-phase operation can increase the energy efficiency and
    throughput of the macro for >5 input bits and >7 output bits. This is
    because the macro's integrator requires 2^N timesteps to process results
    from an N-bit input slice. The 2^N scaling can lead to high integrator
    energy and latency when N is large. Two-phase operation breaks inputs into
    two slices to reduce the effect of this scaling. The two-step operation only
    kicks in when there are five or more input bits.
    """
    EXPECTED = {
        (3, 1, 1): (2.12230988, 43.07875866),
        (4, 2, 1): (1.793083771, 40.33412834),
        (5, 3, 1): (1.165400233, 23.09069279),
        (6, 4, 1): (0.7438677864, 15.95409517),
        (7, 5, 1): (0.4515642, 11.32999012),
        (8, 6, 1): (0.260797508, 9.12235241),
        (3, 1, 0): (2.12230988, 43.07875866),
        (4, 2, 0): (1.793083771, 40.33412834),
        (5, 3, 0): (1.165400233, 23.09069279),
        (6, 4, 0): (0.7438677864, 15.95409517),
        (7, 5, 0): (0.5254091287, 11.03166076),
        (8, 6, 0): (0.4177183338, 8.257196816),
        (9, 7, 0): (0.343873405, 7.750037817),
        (10, 8, 0): (0.2638743409, 6.437388185),
    }
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                OUTPUT_BITS=k[0],
                INPUT_BITS=k[1],
                ONE_PHASE=k[2],
                WEIGHT_BITS=WEIGHT_BITS,
            ),
        )
        for k in EXPECTED
    )
    for e, r in zip(EXPECTED.values(), results):
        r.tops *= 48  # 48 tiles in the full accelerator
        r.add_compare_ref("tops", e[0])
        r.add_compare_ref("tops_per_w", e[1])

    return results


def test_energy_bits_scaling():
    """
    ### Energy Breakdown and Number of Input and Output Bits

    This test replicates the results of Fig. 12 (b) of the paper.

    We show the energy of different components of the macro at varying numbers
    of input and output bits. We will show three categories:

    - Neuron operations and control, which includes the shift_add,
      adc, integrator, and sample components.
    - Input pulses, which includes the row_drivers and cim_unit components.
    - WL (Row) Switching, which includes the wordline_drivers and wordline_cap
      components.

    Note that this macro uses a signed DAC, which is underutilized for 1b
    inputs. For this reason, 1b and 2b inputs required the same number
    of activations for most circuits, and therefore the energy numbers are
    similar for 1b and 2b inputs.

    We see the following:

    - Neuron operations and control energy increases linearly with the number of
      output bits. This is because the variable-precision ADC and control logic
      must process more bits, leading to higher energy consumption.
    - Input pulses energy increases with the number of input bits. Energy
      increases exponentially as the DAC requires 2^N timesteps to process an
      N-bit slice, and there is a jump at 5 input bits, where the macro switches
      to two-phase operation.
    - WL (row) switching energy increases linearly with the number of input
      bits, as the wordlines are used to supply input values.

    """

    EXPECTED = {  # fJ/MAC
        (3, 1, 0): (8.99015248, 10.6186848, 3.446220545),
        (4, 2, 0): (8.270951256, 12.38674172, 3.955661604),
        (5, 3, 0): (15.46307324, 20.11826462, 7.73153662),
        (6, 4, 0): (21.75617998, 29.1084171, 11.50740478),
        (7, 5, 0): (30.92613276, 44.21188287, 14.0246365),
        (8, 6, 0): (37.2192395, 52.48282041, 19.23892768),
        (9, 7, 0): (43.69214654, 61.83258722, 22.83498524),
        (10, 8, 0): (50.52466791, 75.85721686, 27.50986693),
    }
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                OUTPUT_BITS=k[0],
                INPUT_BITS=k[1],
                ONE_PHASE=k[2],
                WEIGHT_BITS=WEIGHT_BITS,
            ),
        )
        for k in EXPECTED
    )
    results.combine_per_component_energy(
        [
            "shift_add",
            "adc",
            "integrator",
            "sample",
            "weight_drivers",  # Just throw this in here. Its basically zero
        ],
        "Neuron ops. and control (1.8V)",
    )
    results.combine_per_component_energy(
        [
            "row_drivers",
            "cim_unit",
        ],
        "Input pulses (Vread=0.5V)",
    )
    results.combine_per_component_energy(
        [
            "wordline_drivers",
            "wordline_cap",
        ],
        "WL Switching (0.9V <-> 2.2V)",
    )
    results.clear_zero_energies()
    for e, r in zip(EXPECTED.values(), results):
        # Scale to fJ/MAC
        # r.per_component_energy = {
        #     k: v / OPS_PER_ARRAY for k, v in r.per_component_energy.items()
        # }
        r.add_compare_ref_energy(
            "WL Switching (0.9V <-> 2.2V)", e[0] * OPS_PER_ARRAY / 1e15
        )
        r.add_compare_ref_energy(
            "Neuron ops. and control (1.8V)", e[1] * OPS_PER_ARRAY / 1e15
        )
        r.add_compare_ref_energy(
            "Input pulses (Vread=0.5V)", e[2] * OPS_PER_ARRAY / 1e15
        )

    return results


def test_area_breakdown():
    """
    ### Area Breakdown

    This test replicates the results of Fig. 11 (f) of the paper.

    We show the area of the macro and its subcomponents. We report the area of:

    - Neurons, which includes the shift add, adc, integrator, and sample
      components.
    - Array drivers, which includes the wordline drivers, row drivers,
      and column drivers.
    - RRAMs, which includes the CiM units.

    """
    results = utl.single_test(utl.quick_run(macro=MACRO_NAME))

    results.combine_per_component_area(
        [
            "shift_add",
            "adc",
            "integrator",
            "sample",
        ],
        "Neurons",
    )
    results.combine_per_component_area(
        [
            "wordline_drivers",
            # 'wordline_cap', # Zero area
            "row_drivers",
            "column_drivers",
        ],
        "Array drivers",
    )
    results.combine_per_component_area(
        [
            "cim_unit",
        ],
        "RRAMs",
    )
    area = 1270 * 256 / 0.28
    results.add_compare_ref_area("Neurons", area * 0.28 * 1e-12)
    results.add_compare_ref_area("Array drivers", area * 0.21 * 1e-12)
    results.add_compare_ref_area("RRAMs", area * 0.12 * 1e-12)
    results.clear_zero_areas()
    return results


def test_array_size_dnn(dnn_name: str):
    """
    ### Exploration of Array Size versus. DNN Energy Efficiency

    In this test, we explore the energy efficiency of the macro at varying array
    sizes for different DNNs. We will set the number of input bits to 8, the
    number of output bits to 10, and we show the area array sizes 64x64,
    128x128, 256x256, 512x512, 1024x1024, and 2048x2048. We will compare the
    energy efficiency of the macro for different DNNs.

    We see that, as the array size increases:

    - Energy for ADC and output processing decreases because the number of array
      rows increases, leading to more analog output reuse and fewer ADC
      converts.
    - Energy for DAC decreases because the number of array columns increases,
      leading to more analog input reuse and fewer DAC converts. Input
      processing energy decreases less than does output processing energy
      because input processing pays energy to drive high-capacitance row wires,
      the capacitance of which increases with the array size. This increase
      cancels some of the energy savings from input reuse.



    increases. This is because the amount of analog output reuse increases,
    leading to lower energy paid for ADC and ADC and digital output sum
    processing. Input processing energy decreases less because input processing
    pays significant energy to supply inputs on high-capacitance bitlines. The
    capacitance of the bitlines increases with the array size, cancelling out
    much of the energy savings from input reuse.

    The energy benefits of larger arrays are strongest for maximum-utilization
    and large-tensor-size workloads. For medium-tensor-size workloads, the
    energy benefits of larger arrays saturate as the array grows larger and
    becomes underutilized for smaller layers. For small-tensor-size workloads,
    underutilization increases energy for all array sizes and leads to a smaller
    array being the lowest-energy choice.
    """
    if dnn_name == "max_utilization":
        layer_paths = [None]
        v = {"MAX_UTILIZATION": True}
    else:
        dnn_dir = utl.path_from_model_dir(f"workloads/{dnn_name}")
        layer_paths = [
            os.path.join(dnn_dir, l) for l in os.listdir(dnn_dir) if l.endswith(".yaml")
        ]
        layer_paths = [
            l for l in layer_paths if "From einsum" not in open(l, "r").read()
        ]
        v = {}

    results = utl.parallel_test(
        utl.delayed(utl.run_layer)(
            macro=MACRO_NAME,
            layer=l,
            variables=dict(
                OUTPUT_BITS=10,
                INPUT_BITS=8,
                ONE_PHASE=0,
                ARRAY_ROWS=x,
                ARRAY_COLS=x,
                **v,
            ),
        )
        for l in layer_paths
        for x in [64, 128, 256, 512, 1024, 2048]
    )

    results.combine_per_component_energy(
        [
            "shift_add",
            "adc",
            "integrator",
            "sample",
        ],
        "ADC and Output Processing",
    )
    results.combine_per_component_energy(
        [
            "row_drivers",
            "cim_unit",
        ],
        "DAC and Analog MAC",
    )
    results.combine_per_component_energy(
        [
            "wordline_drivers",
            "wordline_cap",
        ],
        "Accumulator Control",
    )

    results.clear_zero_energies()

    return results.aggregate_by("ARRAY_ROWS")


if __name__ == "__main__":
    test_tops(),
    test_energy_bits_scaling(),
    test_area_breakdown(),
    test_array_size_dnn("max_utilization"),
