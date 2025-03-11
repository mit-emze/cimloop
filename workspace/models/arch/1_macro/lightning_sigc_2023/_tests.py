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
    This test verifies the energy breakdown of the accelerator.
    """
    results = utl.parallel_test(
        [utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                MAX_UTILIZATION=False
            ),
        )]
    )

    def w2pj(*args):  # * 97GHz * 1e12 J->pJ
        return [y * 0.01030927e-9 * 1e12 for y in args]
    
    for r in results:
        r.per_component_energy *= 1e12  # J -> pJ

    # results.combine_per_component_energy()
    results.add_compare_ref_energy("laser", w2pj(3.88 * 1e-6))
    results.add_compare_ref_energy("photodetector", w2pj(3.88 * 1e-6))
    results.add_compare_ref_energy("individual_modulator_placeholder", w2pj(3.88 * 1e-6))
    results.add_compare_ref_energy("WMUs", w2pj(3.88 * 1e-6))
    results.add_compare_ref_energy("adc", w2pj(0.075))
    results.add_compare_ref_energy("input_dac", w2pj(0.077))
    results.add_compare_ref_energy("weight_dacs", w2pj(0.077))
    results.add_compare_ref_energy("memory_controller", w2pj(0.0186))
    results.add_compare_ref_energy("packet_io_input", w2pj(0.009))
    results.add_compare_ref_energy("packet_io_output", w2pj(0.009))

    return results


def test_area_breakdown():
    """
    This test verifies the area breakdown of the accelerator.
    """
    results = utl.single_test(utl.quick_run(macro=MACRO_NAME))

    total_area = 2095.787 * 1000000  # um^2
    expected_area = {
        "Packet I/O": 0.00009876957916047766 * total_area * 2, # x2 because we have both input and output packer IO
        "Memory Controller": 0.03549979077072240642 * total_area,
        "DAC": 0.16604740844370157845 * total_area,
        "ADC": 0.00664189633774806313 * total_area,
        "input_modulator": 0.02862886352477613421 * total_area,
        "weight_modulator": 0.68709272459462722118 * total_area,
        "Photodetector": 0.00000036644945311713 * total_area,
        "Laser": 0.00000477147725412935 * total_area
    }

    for r in results:
        r.per_component_area *= 1e12  # m^2 -> um^2

    results.combine_per_component_area(["packet_io_input", "packet_io_output"], "Packet I/O")
    results.add_compare_ref_area("Packet I/O", [expected_area["Packet I/O"]])
    results.combine_per_component_area(["memory_controller"], "Memory Controller")
    results.add_compare_ref_area("Memory Controller", [expected_area["Memory Controller"]])
    results.combine_per_component_area(["weight_dacs", "input_dac"], "DAC")
    results.add_compare_ref_area("DAC", [expected_area["DAC"]])
    results.combine_per_component_area(["adc"], "ADC")
    results.add_compare_ref_area("ADC", [expected_area["ADC"]])
    results.combine_per_component_area(["WMUs"], "weight_modulator")
    results.add_compare_ref_area("weight_modulator", [expected_area["weight_modulator"]])
    results.combine_per_component_area(["individual_modulator_placeholder"], "input_modulator")
    results.add_compare_ref_area("input_modulator", [expected_area["input_modulator"]])
    results.combine_per_component_area(["photodetector"], "Photodetector")
    results.add_compare_ref_area("Photodetector", [expected_area["Photodetector"]])
    results.combine_per_component_area(["laser"], "Laser")
    results.add_compare_ref_area("Laser", [expected_area["Laser"]])

    results.clear_zero_areas()
    return results


def test_full_dnn(dnn_name: str, batch_sizes: list, num_parallel_wavelengths: list, num_parallel_batches: list, num_parallel_weights: list):
    """
    This test evaluates full-DNN energy efficiency and throughput.
    """
    dnn_dir = utl.path_from_model_dir(f"workloads/{dnn_name}")
    layer_paths = [
        os.path.join(dnn_dir, l) for l in os.listdir(dnn_dir) if l.endswith(".yaml")
    ]

    def callfunc(spec):  # Speed up the test by reducing the victory condition
        spec.mapper.victory_condition = 10

    results = utl.parallel_test(
        utl.delayed(utl.run_layer)(
            macro=MACRO_NAME,
            layer=l,
            variables=dict(
                BATCH_SIZE=n,
                NUM_WAVELENGTHS=w,
                NUM_PARALLEL_WEIGHTS=pw,
                PARALLEL_BATCH_SIZE=b,
                SCALING=f'"{s}"',
            ),
            system="ws_dummy_buffer_one_macro",
            callfunc=callfunc,
        )
        for s in ["conservative"]
        for n in batch_sizes
        for l in layer_paths
        for w in num_parallel_wavelengths
        for pw in num_parallel_weights
        for b in num_parallel_batches
    )
    return results

if __name__ == "__main__":
    test_energy_breakdown()
    test_area_breakdown()
    test_full_dnn("alexnet")
    test_full_dnn("vgg16")
