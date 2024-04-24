import sys
import os

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl
import scripts
# fmt: on


N_COLUMNS = 256
N_ADC_USES = N_COLUMNS * 1
N_OUTPUTS = N_COLUMNS / 1


def test_tops():
    """
    ### Energy Efficiency, Throughput, and Compute Density

    This test replicates the results of Table II in the paper.

    We show the area, energy efficiency, and throughput of the macro at 0.85V
    and 1.2V power supplies using 1b inputs and weights.

    We see that increasing the voltage from 0.85V to 1.2V increases throughput
    and compute density at the cost of increased the energy consumption.

    Modeled compute density is higher than reported because we did not model
    the area of the control logic that manages the sparsity controller.
    """
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                VOLTAGE=x,
                INPUT_BITS=1,
                OUTPUT_BITS=1,
                WEIGHT_BITS=1,
            ),
        )
        for x in [0.85, 1.2]
    )

    components_for_tops = [
        "row_drivers",
        "cim_unit",
        "adc",
        "column_drivers",
        "bitcell_capacitor",
        "weight_drivers",
    ]
    results.combine_per_component_energy(components_for_tops, "TOPS Components")
    results.combine_per_component_area(components_for_tops, "TOPS Components")

    for r, tops, tops_per_w_1b, tops_per_mm2_1b in zip(
        results,
        [0.874, 2.185],  # Expected 1b TOPS
        [400, 192],  # Expected 1b TOPS/W
        [0.24, 0.6],  # Expected 1b TOPS/mm^2
    ):
        r.tops_per_w_1b = (
            2 * r.computes / r.per_component_energy["TOPS Components"] / 1e12
        )
        r.tops_per_mm2_1b = r.tops / r.per_component_area["TOPS Components"] / 1e6

        r.add_compare_ref("tops_1b", tops)
        r.add_compare_ref("tops_per_w_1b", tops_per_w_1b)
        r.add_compare_ref("tops_per_mm2_1b", tops_per_mm2_1b)

    return results


def test_area_breakdown():
    """
    ### Area Breakdown

    This test replicates the results of Fig. 11 in the paper.

    We show the area of the ADC, CiM, NMC data path, and sparsity controller.
    Modeled sparsity controller area is lower than reference area due to
    additional control logic that we did not model.
    """
    results = utl.single_test(utl.quick_run(macro=MACRO_NAME))
    results.combine_per_component_area(["adc"], "ADC")
    results.add_compare_ref_area("ADC", [0.497e-6])

    results.combine_per_component_area(
        [
            "row_drivers",
            "weight_drivers",
            "cim_unit",
            "column_drivers",
            "bitcell_capacitor",
        ],
        "CiM",
    )
    results.add_compare_ref_area("CiM", [2.91e-6])

    results.combine_per_component_area(["out_datapath", "shift_add"], "NMC Data Path")
    results.add_compare_ref_area("NMC Data Path", [0.497e-6])

    results.combine_per_component_area(["input_zero_gating"], "Sparsity Controller")
    results.add_compare_ref_area("Sparsity Controller", [0.392e-6])

    return results


def test_energy_breakdown():
    """
    ### Energy Breakdown

    This test replicates the results of Table I in the paper.

    We show the area and energy of the macro at 0.85V and 1.2V power supplies
    using 1b inputs and weights. We will report the energy of the ADC, CiM,
    and NMC data path.

    We see that increasing the voltage from 0.85V to 1.2V increases the energy
    consumption of each component of the macro.
    """
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                VOLTAGE=x,
                INPUT_BITS=1,
                OUTPUT_BITS=1,
                WEIGHT_BITS=1,
            ),
        )
        for x in [0.85, 1.2]
    )

    results.add_compare_ref_energy(
        "adc", [1.79e-12 * N_ADC_USES, 3.56e-12 * N_ADC_USES]
    )
    results.combine_per_component_energy(
        [
            "row_drivers",
            "weight_drivers",
            "cim_unit",
            "bitcell_capacitor",
            "column_drivers",
        ],
        "CiM",
    )
    results.add_compare_ref_energy("CiM", [9.7e-12 * N_ADC_USES, 20.4e-12 * N_ADC_USES])
    results.combine_per_component_energy(["out_datapath", "shift_add"], "NMC Data Path")
    results.add_compare_ref_energy(
        "NMC Data Path", [8.3e-12 * N_OUTPUTS, 14.7e-12 * N_OUTPUTS]
    )
    return results


def test_tops_bits_scaling():
    """
    ### Throughput versus Number of Input and Weight Bits

    This test replicates the results of Fig. 10 C_cimu in the paper.

    We show the area and throughput of the macro with 1, 2, 4, and 8b inputs
    and weights. For each configuration, we measure the throughput of the macro
    in TOPS.

    We see that the this macro can flexibly trade off bit precision and
    throughput. It does so by computing with 1b slices of inputs and
    weights, then shifting + adding together results from a variable number of
    slices. Each input slice is processed in a different timestep, so fewer
    (more) input slices will decrease (increase) latency. Each weight slice is
    stored in a different column, so fewer (more) weight slices will decrease
    (increase) storage density.

    There is a slight discrepancy at the highest-bitwidth result, where the
    throughput of the published macro decreases more than expected. This is
    due to bottlenecks in the published macro's memory hierarchy.
    """

    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                INPUT_BITS=n_bits,
                OUTPUT_BITS=1,
                WEIGHT_BITS=1,
            ),
        )
        for n_bits in [1, 2, 4, 8]
    )

    expected = [0.874 / x for x in [1, 2, 4, 600 / 64]]
    results.add_compare_ref("tops", expected)
    return results


def test_column_folding_dnn(dnn_name: str = "resnet18"):
    """
    ### Exploration of Inter-Column Output Reuse Versus Energy Efficiency

    This test explores how the macro's column folding strategy impacts the
    energy efficiency and throughput of a DNN. Column folding connects columns
    together to share outputs, rather than inputs as is done in non-folded
    columns.

    We can see that, as the number of folded columns increases, the energy due
    to output processing (column readout, ADC) decreases because folding columns
    increases output reuse and allows output readout circuitry energy to be
    amortized across more computations (*e.g.,* one ADC can read the results
    from more than column at a time instead of just one). However, the energy
    due to input processing (DAC, row drivers) increases because amount of
    input reuse decreases leading to more DAC converts and input driver
    activations.
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
            layer=p,
            variables=dict(
                N_FOLDED_COLUMNS=x,
                N_NON_FOLDED_COLUMNS=192 // x,
                **v,
            ),
            system="ws_dummy_buffer_many_macro",
        )
        for p in layer_paths
        for x in range(1, 9)
    )

    results.combine_per_component_energy(
        ["adc", "column_drivers", "out_datapath", "shift_add"], "Output Processing"
    )
    results.combine_per_component_energy(
        ["row_drivers", "input_zero_gating"], "Input Processing"
    )
    results.combine_per_component_energy(["cim_unit", "bitcell_capacitor"], "Other")
    results.clear_zero_energies()

    return results.aggregate_by("N_FOLDED_COLUMNS")


if __name__ == "__main__":
    test_area_breakdown(),
    test_tops(),
    test_energy_breakdown(),
    test_tops_bits_scaling(),
    test_column_folding_dnn("resnet18"),
