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
    This test explores the energy, area, and latency of the accelerator
    computing MVM operations. We note a few differences from the original ISAAC
    paper. Notably, we made a few changes to the quantization, and we use
    data-value-dependent models while ISAAC used a simple fixed-power model.
    
    We note:
    - Energy is dominated by the ADC and memory cells due to the high ADC precision
      and large number of slices.
    - Area is dominated by ADC.
    """
    results = utl.parallel_test(
        [utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            tile="isaac",
            chip="large_router",
        )]
    )
    results.clear_zero_areas()
    results.clear_zero_energies()
    return results

def test_full_dnn(dnn_name: str):
    """
    This test explores the energy, area, and latency of the accelerator when
    running full DNN workloads.
    """
    dnn_dir = utl.path_from_model_dir(f"workloads/{dnn_name}")
    layer_paths = [
        os.path.join(dnn_dir, l) for l in os.listdir(dnn_dir) if l.endswith(".yaml")
    ]

    layer_paths = [l for l in layer_paths if "From einsum" not in open(l, "r").read()]

    results = utl.parallel_test(
        utl.delayed(utl.run_layer)(
            macro=MACRO_NAME,
            layer=l,
            tile="isaac",
            chip="large_router",
        )
        for l in layer_paths
    )
    results.clear_zero_energies()
    return results


if __name__ == "__main__":
    test_energy_breakdown()
    test_full_dnn("resnet18")
