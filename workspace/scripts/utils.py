import shutil
import time
from typing import Callable, Union, Iterable, List
import os
import threading
import joblib
import pytimeloop.timeloopfe.v4 as tl
import sys
import importlib.util
import sys
from tqdm import tqdm

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_SCRIPT_DIR)
from processors import ArrayProcessor
from tl_output_parsing import parse_timeloop_output, MacroOutputStats, MacroOutputStatsList

from plots import *
# fmt: on

from joblib import delayed as delayed


def single_test(result) -> MacroOutputStatsList:
    return MacroOutputStatsList([result])


def parallel_test(
    delayed_calls: List[Callable], n_jobs: int = 32
) -> MacroOutputStatsList:
    if not isinstance(delayed_calls, Iterable):
        delayed_calls = [delayed_calls]

    delayed_calls = list(delayed_calls)
    return MacroOutputStatsList(
        tqdm(
            joblib.Parallel(return_as="generator", n_jobs=n_jobs)(delayed_calls),
            total=len(delayed_calls),
        )
    )


def path_from_model_dir(*args):
    return os.path.abspath(os.path.join(THIS_SCRIPT_DIR, "..", "models", *args))


def get_run_dir():
    out_dir = os.path.join(
        THIS_SCRIPT_DIR,
        "..",
        "outputs",
        f"{os.getpid()}.{threading.current_thread().ident}",
    )
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_spec(
    macro: str,
    tile: str = None,
    chip: str = None,
    system: str = "ws_dummy_buffer_one_macro",
    iso: str = None,
    dnn: str = None,
    layer: str = None,
    max_utilization: bool = False,
    extra_print: str = "",
    jinja_parse_data: dict = None,
) -> tl.Specification:
    paths = [
        os.path.abspath(
            os.path.join(THIS_SCRIPT_DIR, "..", "models", "top.yaml.jinja2")
        )
    ]

    jinja_parse_data = {
        **(jinja_parse_data or {}),
        "macro": macro,
        "tile": tile,
        "chip": chip,
        "system": system,
        "iso": iso if iso else macro,
        "dnn": dnn,
        "layer": layer,
    }
    jinja_parse_data = {k: v for k, v in jinja_parse_data.items() if v is not None}

    paths2print = [p for p in paths]
    while any(paths2print):
        if all(paths2print[0][0] == p[0] for p in paths2print):
            paths2print = [p[1:] for p in paths2print]
        else:
            break
    paths2print = ", ".join(paths2print)

    if not extra_print:
        extra_print = f"{os.getpid()}.{threading.current_thread().ident}"

    spec = tl.Specification.from_yaml_files(
        *paths, processors=[ArrayProcessor], jinja_parse_data=jinja_parse_data
    )
    if max_utilization:
        spec.variables["MAX_UTILIZATION"] = True

    return spec


def run_mapper(
    spec: tl.Specification,
    accelergy_verbose: bool = False,
) -> dict:
    output_dir = get_run_dir()

    run_prefix = f"{output_dir}/timeloop-mapper"
    mapper_result = tl.call_mapper(
        specification=spec,
        output_dir=output_dir,
        log_to=os.path.join(output_dir, f"{run_prefix}.log"),
    )
    if accelergy_verbose:
        tl.call_accelergy_verbose(
            specification=spec,
            output_dir=output_dir,
            log_to=os.path.join(output_dir, "accelergy.log"),
        )

    return MacroOutputStats.from_output_stats(mapper_result)


def quick_run(
    macro: str,
    variables: dict = None,
    accelergy_verbose: bool = False,
    **kwargs,
):
    spec = get_spec(
        macro=macro,
        system="ws_dummy_buffer_one_macro",
        max_utilization=True,
        **kwargs,
    )
    variables = variables or {}
    spec.variables.update(variables)
    for k in list(spec.variables.keys()):
        if k not in variables:
            spec.variables[k] = spec.variables.pop(k)

    return run_mapper(spec, accelergy_verbose=accelergy_verbose)


def get_diagram(
    macro: str,
    container_names: Union[str, List[str]] = (),
    ignore: List[str] = (),
    variables: dict = None,
    **kwargs,
):
    spec = get_spec(
        macro=macro,
        system="ws_dummy_buffer_one_macro",
        max_utilization=True,
        **kwargs,
    )
    spec.variables.update(variables or {})
    return spec.to_diagram(container_names, ignore)


def get_test(
    macro: str,
    function_name: str,
):
    # Python path is macro path + _tests.py
    path = os.path.abspath(
        os.path.join(
            THIS_SCRIPT_DIR, "..", "models", "arch", "1_macro", macro, "_tests.py"
        )
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"No test file found at {path}")
    modspec = importlib.util.spec_from_file_location("modname", path)
    module = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(module)
    return getattr(module, function_name)


def run_layer(
    macro: str,
    layer: str,
    variables: dict = None,
    callfunc: Callable = None,
    iso: str = None,
    tile=None,
    chip=None,
    system="ws_dummy_buffer_many_macro",
):
    spec = get_spec(
        macro=macro, iso=iso, layer=layer, tile=tile, chip=chip, system=system
    )
    spec.architecture.name2leaf("macro").attributes["has_power_gating"] = True

    variables = variables or {}

    spec.variables.update(variables)
    for k in list(spec.variables.keys()):
        if k not in variables:
            spec.variables[k] = spec.variables.pop(k)

    if callfunc is not None:
        callfunc(spec)

    try:
        return run_mapper(spec=spec)
    except Exception as e:
        print(f"Error processing spec with {macro}, {iso}, {layer}, {variables}")
        raise e
