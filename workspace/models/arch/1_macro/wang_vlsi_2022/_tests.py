import sys
import os

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl
import scripts
# fmt: on


TOPS_PER_WATT = 16.37572276
PJ_PER_MVM = 2 / TOPS_PER_WATT * 16 * 64 * 8
TOTAL_AREA = 0.124e6  # um^2


def test_energy_breakdown():
    """
    ### Energy Breakdown

    This test replicates the results of Fig. 22(a) of the paper.

    We show the energy breakdown of the macro. The energy is broken down into
    the following components:

    - ADC: Energy consumed by the ADC
    - DAC: Energy consumed by the DAC
    - MAC: Energy consumed by the MAC, including the row drivers, select
      wordline drivers, CiM unit, and C-2C multiplier.
    - Misc: The weight drivers are miscellaneous components in our model, but
      they consume no energy in this weight-stationary test. Misc also includes
      control circuitry in the reference.

    Modeled miscellaneous energy is lower than reference because we do not model
    the control circuitry.

    """
    results = utl.single_test(utl.quick_run(macro=MACRO_NAME))
    results.combine_per_component_energy(["adc"], "ADC")
    results.add_compare_ref_energy("ADC", PJ_PER_MVM * 0.34 * 1e-12)

    results.combine_per_component_energy(["dac"], "DAC")
    results.add_compare_ref_energy("DAC", PJ_PER_MVM * 0.22 * 1e-12)

    results.combine_per_component_energy(
        [
            "row_drivers",
            "select_wordline_drivers",
            "cim_unit",
            "c2c_multiplier_digital_port",
            "c2c_multiplier_analog_port",
            "column_drivers",
        ],
        "MAC",
    )
    results.add_compare_ref_energy("MAC", PJ_PER_MVM * 0.4 * 1e-12)

    results.combine_per_component_energy(["weight_drivers"], "Misc")
    results.add_compare_ref_energy("Misc", PJ_PER_MVM * 0.04 * 1e-12)

    return results


def test_area_breakdown():
    """
    ### Area Breakdown

    This test replicates the results of Fig. 22(b) of the paper.

    We show the area breakdown of the macro. The area is broken down into the
    following components:

    - ADC: Area consumed by the ADC
    - DAC: Area consumed by the DAC
    - MAC: Area consumed by the MAC, including the row drivers, select wordline
      drivers, CiM unit, and C-2C multiplier.
    - Misc: Area consumed by the weight drivers and control circuitry.

    Modeled miscellaneous area is lower than reference because we do not model
    the control circuitry in the weight drivers.

    """
    results = utl.single_test(utl.quick_run(macro=MACRO_NAME))
    results.combine_per_component_area(["adc"], "ADC")
    results.add_compare_ref_area("ADC", [0.13 * TOTAL_AREA * 1e-12])
    results.combine_per_component_area(["dac"], "DAC")
    results.add_compare_ref_area("DAC", [0.3 * TOTAL_AREA * 1e-12])
    results.combine_per_component_area(
        [
            "select_wordline_drivers",
            "cim_unit",
            "c2c_multiplier_digital_port",
            "row_drivers",
            "column_drivers",
        ],
        "MAC",
    )
    results.add_compare_ref_area("MAC", [0.46 * TOTAL_AREA * 1e-12])
    results.combine_per_component_area(["weight_drivers"], "Misc")
    results.add_compare_ref_area("Misc", [0.11 * TOTAL_AREA * 1e-12])
    return results


def test_voltage_scaling():
    """
    ### Voltage Scaling

    This test replicates the results of Fig. 23 of the paper.

    We show the effects of voltage scaling on the energy efficiency and
    throughput of the macro, testing supply voltages of 0.7V, 0.8V, 0.9V, 1V,
    and 1.1V.

    We can see that increasing the supply voltage increases throughput and
    compute density at the cost of lower energy efficiency.

    Modeled and reference compute density varies because we did not model the
    area of some miscellaneous components, leading to the model having a smaller
    area and higher compute density. This could be corrected by adding
    additional components to the model. We also use a different scaling factor
    for voltage versus energy, leading to a different curve shape. This could be
    corrected by adjusting the VOLTAGE_ENERGY_SCALE formula and propagating the
    value to each subcomponent model.
    """
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(macro=MACRO_NAME, variables=dict(VOLTAGE=x))
        for x in [0.7, 0.8, 0.9, 1, 1.1]
    )

    for r, tops_mm, tops_w in zip(
        results,
        [2.377, 2.858, 3.200, 3.596, 3.941],
        [31.998, 22.590, 18.439, 16.376, 15.467],
    ):
        r.add_compare_ref("tops_per_mm2", tops_mm)
        r.add_compare_ref("tops_per_w", tops_w)
        r.add_compare_ref("tops", tops_mm * TOTAL_AREA / 1e6)
    return results


def test_tops():
    """
    ### Energy Efficiency, Throughput, and Compute Density

    This test replicates the results of Table III in the paper.

    In this test, we show the energy efficiency, throughput, and compute density
    of the macro at 0.7V and 1.1V supply voltages.

    We see that increasing the supply voltage increases throughput at the cost of
    lower energy efficiency.
    """
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(macro=MACRO_NAME, variables=dict(VOLTAGE=x))
        for x in [0.7, 1.1]
    )
    for r, tops_mm, tops_w, tops in zip(
        results,
        [2.4, 4.0],
        [32.2, 15.5],
        [0.3, 0.5],
    ):
        r.add_compare_ref("tops_per_mm2", tops_mm)
        r.add_compare_ref("tops_per_w", tops_w)
        r.add_compare_ref("tops", tops)
    return results


def test_full_system_dnn(dnn_name: str, batch_size: int = None):
    """
    ### Exploration of Full-System Energy Efficiency

    In this test, we look at the full-system energy breakdown when running DNNs
    on a CiM accelerator. We place the macro in a chip with local input/output
    buffers, routers for on-chip data movement, a global buffer, and DRAM. We
    show the area and energy spent on DRAM, the global buffer, and other
    components.

    We compare three scenarios:

    1. Inputs, outputs, and weights stored off-chip in DRAM and fetched for each
       layer
    2. Inputs and outputs fetched from DRAM, weights stationary (pre-loaded for
       each layer)
    3. Weights stationary, layers fused to keep inputs/outputs on-chip in the
       global

    We can see that weight-stationary processing significantly reduces overall
    energy due to fewer weight fetches from off-chip. Benefits are limited,
    however, because inputs and outputs still must be fetched from off-chip. To
    see further benefits, fusing layers is necessary to keep data on-chip
    between DNN layers. We note that weight-stationary CiM requires sufficient
    memory to keep all DNN weights on-chip. To store large DNNs, this may
    require a multi-chip pipeline or dense storage technologies.
    """

    dnn_dir = utl.path_from_model_dir(f"workloads/{dnn_name}")
    layer_paths = [
        os.path.join(dnn_dir, l) for l in os.listdir(dnn_dir) if l.endswith(".yaml")
    ]

    layer_paths = [l for l in layer_paths if "From einsum" not in open(l, "r").read()]

    if "gpt2_medium" in dnn_name:
        layer_paths = layer_paths[:-1]

    def callfunc(spec):
        spec.architecture.find("shared_router_group").spatial.meshX = 64
        spec.architecture.find("shared_router_group").attributes.has_power_gating = True
        spec.architecture.find("shared_router_group").constraints.spatial.no_reuse = []

        spec.architecture.find("tile_in_chip").spatial.meshX = 16
        spec.architecture.find("tile_in_chip").attributes.has_power_gating = True
        spec.architecture.find("tile_in_chip").constraints.spatial.no_reuse = []

        if batch_size is not None:
            spec.problem.instance["N"] = batch_size
        spec.architecture.find("output_buffer").constraints.temporal.iter_only = []

    results = utl.parallel_test(
        utl.delayed(utl.run_layer)(
            macro=MACRO_NAME,
            layer=l,
            variables=dict(EXPERIMENT_NAME=s),
            tile="input_output_bufs",
            chip="large_router_glb",
            system=system,
            callfunc=callfunc,
        )
        for l in layer_paths
        for s, system in (
            ('"All Tensors Off-Chip"', "fetch_all_lpddr4"),
            ('"Weight-Stationary"', "fetch_weights_lpddr4"),
            ('"Weight-Stationary + Fusion"', None),
        )
    )

    for r in results:
        r.per_component_energy.setdefault("main_memory", 0)

    results.combine_per_component_energy(
        [
            "c2c_multiplier_analog_port",
            "c2c_multiplier_digital_port",
            "cim_unit",
            "adc",
            "select_wordline_drivers",
            "row_drivers",
            "dac",
            "output_buffer",
            "input_buffer",
            "router",
            "weight_drivers",
            "column_drivers",
        ],
        "Macro & Other On-Chip Data Movement",
    )
    results.combine_per_component_energy(["glb"], "Global Buffer")
    results.combine_per_component_energy(["main_memory"], "Off-Chip DRAM")
    results.clear_zero_energies()

    return results


if __name__ == "__main__":
    test_energy_breakdown(),
    test_area_breakdown(),
    test_voltage_scaling(),
    test_tops(),
    test_full_system_dnn("resnet18")
