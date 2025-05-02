import sys
import os

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl
import scripts
# fmt: on

def test_area_energy_breakdown():
    """
    ### Area and energy breakdown
    This example architecture doesn't have a suite of tests, but you may
    look at the other example architectures for inspiration.        
    """
    # results = utl.single_test(utl.quick_run(macro=MACRO_NAME))
    layer_path = utl.path_from_model_dir(f"workloads/llm/gemm.yaml")
    print("Running on layer", layer_path)

    results = utl.parallel_test(
        utl.delayed(utl.run_layer)(
            macro=MACRO_NAME,
            system="_none",
            layer=layer_path,
            variables=dict(
                QUANTIZE=q,
            
            ),
        )
        for q in [False, True]
    )
    results.clear_zero_areas()
    results.clear_zero_energies()
    return results

if __name__ == "__main__":
    test_area_energy_breakdown()
