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
    computing MVM operations with and without speculation.

    We find that speculation increases array and DAC energy due to speculation
    cycles. Increases the input buffer energy due to 2× fetches of inputs (for
    speculation & recovery cycles). It decreases output buffer energy and ADC
    energy due to fewer analog outputs being read from the array and written to
    the output buffer.
    """
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro=MACRO_NAME,
            variables=dict(
                SPECULATION_ENABLED=s,
            ),
            tile="raella",
        )
        for s in [False, True]
    )
    results.clear_zero_areas()
    results.clear_zero_energies()
    return results


def test_full_dnn(dnn_name: str):
    """
    This test explores the energy, area, and latency of the accelerator when
    running full DNN workloads with and without speculation.

    We can observe the following:
    - Speculation generally reduces energy and increases latency due to the factors
      described in the previous test.
    - RAELLA is more energy efficient for layers with more input channels due to
      higher utilization of arrays (which leads to lower ADC energy, per the
      Titanium Law described in the paper).
    - Energy and latency can also increase for layers with signed inputs (e.g., many
      Transformer layers) because RAELLA processes signed inputs in two different
      cycles, which increases the number of array activations and ADC uses.

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
            variables=dict(
                SPECULATION_ENABLED=s,
            ),
            tile="raella",
        )
        for l in layer_paths
        for s in [False, True]
    )
    results.clear_zero_energies()
    return results


if __name__ == "__main__":
    test_energy_breakdown()
    test_full_dnn("resnet18")
