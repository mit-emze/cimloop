import os
from typing import Any, Iterable, Tuple, List, Union
import timeloopfe.v4 as tl
import yaml


def parse_stats_file(path: str) -> Tuple[int, int, float, dict]:
    """!@brief Read the stats from a file.
    !@param path The path to the stats file.
    !@returns A tuple of (cycles, computes, energy).
    """
    lines = open(path, "r").readlines()
    cycles, computes, util, energy = None, None, None, {}
    for i, l in enumerate(lines):
        if "Computes =" in l:
            computes = int(l.split()[-1])
            break
        if "Cycles: " in l:
            cycles = int(l.split()[-1])
        if "Utilization" in l:
            util = float(l.split()[-1][:-1]) / 100

    assert cycles is not None, f"Could not find cycles in stats at {path}."
    assert computes is not None, f"Could not find computes in stats at {path}."
    assert util is not None, f"Could not find utilization in stats at {path}."

    for l in lines[i + 1 :]:
        if "=" in l:
            e, v = l.rsplit("=", 1)
            if e.strip() == "Total":
                continue
            energy[e.strip()] = float(v.strip()) * computes / 1000

    return cycles, computes, util, energy


def get_area_from_art(path: str) -> dict:
    """!@brief Get the area from an ART file.
    !@param path The path to the ART file.
    !@returns A dictionary of {name: area}.
    """
    d = yaml.load(open(path, "r").read(), Loader=yaml.SafeLoader)
    name2area = {}
    for x in d["ART"]["tables"]:
        namecount = x["name"].split(".", 1)[1]
        name = namecount.split("[", 1)[0]
        count = int(namecount.split("[", 1)[1].split(".")[-1][:-1])
        name2area[name] = count * x["area"]
    return name2area


def get_area_from_art_verbose(path: str) -> dict:
    """!@brief Get the area from an ART file.
    !@param path The path to the ART file.
    !@returns A dictionary of {name: area}.
    """
    d = yaml.load(open(path, "r").read(), Loader=yaml.SafeLoader)
    name2area = {}
    for x in d["ART_summary"]["table_summary"]:
        namecount = x["name"].split(".", 1)[1]
        name = namecount.split("[", 1)[0]
        count = int(namecount.split("[", 1)[1].split(".")[-1][:-1])
        if isinstance(x["primitive_estimations"], str):
            name2area[name] = count * x["area"]
        else:
            for sub in x["primitive_estimations"]:
                sub_name = sub["name"].split("[")[0]
                sub_total_area = sub["total_component_area"]
                name2area[f"{name}.{sub_name}"] = count * sub_total_area
    return name2area


class TestOutput:
    name: str
    utilization: float
    computes: int
    cycles: int
    cycle_seconds: float
    energy: dict
    area: dict
    variables: dict
    tops: float
    tops_per_mm2: float
    tops_per_w: float
    macs: int
    macs_1b: int
    tops_1b: float
    tops_per_mm2_1b: float
    tops_per_w_1b: float
    mapping: str

    def __init__(
        self,
        name: str,
        utilization: float,
        computes: int,
        cycles: int,
        cycle_seconds: float,
        energy: dict,
        area: dict,
        variables: dict,
        mapping: str,
    ):
        self.name = name
        self.utilization = utilization
        self.computes = computes
        self.cycles = cycles
        self.cycle_seconds = cycle_seconds
        self.latency = cycles * cycle_seconds
        self.energy = energy
        self.area = area
        # Get rid of all variables that result in a Callable. These can't pickle.
        self.variables = {k: v for k, v in variables.items() if not callable(v)}
        self.mapping = mapping

        self.input_bits = variables["INPUT_BITS"]
        self.weight_bits = variables["WEIGHT_BITS"]
        self.output_bits = variables["OUTPUT_BITS"]
        self.encoded_input_bits = variables["ENCODED_INPUT_BITS"]
        self.encoded_weight_bits = variables["ENCODED_WEIGHT_BITS"]
        self.encoded_output_bits = variables["ENCODED_OUTPUT_BITS"]

        computes /= self.encoded_output_bits
        computes /= self.encoded_input_bits
        computes /= self.encoded_weight_bits

        self.macs = computes
        self.tops = self.macs / (cycles * cycle_seconds) / 1e12 * 2
        self.total_area = sum(area.values())
        self.total_energy = sum(energy.values())
        self.tops_per_mm2 = self.tops / self.total_area * 1e6
        self.tops_per_w = self.macs / self.total_energy * 2

        mult_1b = self.input_bits * self.weight_bits
        self.macs_1b = self.macs * mult_1b
        self.tops_1b = self.tops * mult_1b
        self.tops_per_mm2_1b = self.tops_per_mm2 * mult_1b
        self.tops_per_w_1b = self.tops_per_w * mult_1b

    @staticmethod
    def aggregate(tests: List["TestOutput"]):
        results = {}

        for grab_last in ["name", "cycle_seconds", "area", "variables"]:
            results[grab_last] = getattr(tests[-1], grab_last)

        for to_sum in ["computes", "cycles"]:
            results[to_sum] = sum(getattr(t, to_sum) for t in tests)

        # Sum
        results["energy"] = {}
        for t in tests:
            for k, v in t.energy.items():
                results["energy"][k] = results["energy"].get(k, 0) + v

        # Weighted average
        results["utilization"] = (
            sum(t.utilization * t.computes for t in tests) / results["computes"]
        )

        results["mapping"] = None

        return TestOutput(**results)

    def access(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            if key in self.variables:
                return self.variables[key]
            raise AttributeError(
                f"Could not find key {key} in {self.name}. Available keys: "
                f"{self.__annotations__} and {list(self.variables.keys())}"
            )

    @staticmethod
    def aggregate_by(tests: List["TestOutput"], *keys: Union[List[str], str]):
        to_agg = {}
        for t in tests:
            key = tuple(t.access(k) for k in keys)
            to_agg[key] = to_agg.get(key, []) + [t]

        return TestOutputList(TestOutput.aggregate(v) for v in to_agg.values())

    def consolidate_area(self, from_keys: List[str], to: str):
        if not all(k in self.area for k in from_keys):
            raise KeyError(
                f"Could not find all keys {from_keys} in area for {self.name}. "
                f"Keys: {self.area.keys()}"
            )
        assert len(set(from_keys)) == len(from_keys), (
            f"Duplicate keys found in {from_keys}. " f"Keys: {self.area.keys()}"
        )
        self.area[to] = sum(self.area.pop(k) for k in from_keys)

    def consolidate_energy(self, from_keys: List[str], to: str):
        if not all(k in self.energy for k in from_keys):
            raise KeyError(
                f"Could not find all keys {from_keys} in energy for {self.name}. "
                f"Keys: {self.energy.keys()}"
            )
        assert len(set(from_keys)) == len(from_keys), (
            f"Duplicate keys found in {from_keys}. " f"Keys: {self.energy.keys()}"
        )
        # print(f"Consolidating {from_keys} to {to} for {self.name}")
        # for k in from_keys:
        #     print(f"\t Adding {k} with energy {self.energy[k]}")
        self.energy[to] = sum(self.energy.pop(k) for k in from_keys)
        # print(f"\t {to} energy: {self.energy[to]}")

    def consolidate_area_energy(self, from_keys: List[str], to: str):
        self.consolidate_area(from_keys, to)
        self.consolidate_energy(from_keys, to)

    def add_compare_ref(self, name: str, reference_value: Any):
        setattr(
            self, name, Comparison(reference=reference_value, model=getattr(self, name))
        )

    def add_compare_ref_area(self, name: str, reference_value: Any):
        self.area[name] = Comparison(reference=reference_value, model=self.area[name])

    def add_compare_ref_energy(self, name: str, reference_value: Any):
        self.energy[name] = Comparison(
            reference=reference_value, model=self.energy[name]
        )

    def get_compare_ref_area(self, per_mac: bool = False):
        return {k: v for k, v in self.area.items() if isinstance(v, Comparison)}

    def get_compare_ref_energy(self, per_mac: bool = False):
        return {k: v for k, v in self.energy.items() if isinstance(v, Comparison)}

    def clear_zero_energies(self):
        for k in list(self.energy.keys()):
            if self.energy[k] == 0:
                del self.energy[k]

    def clear_zero_areas(self):
        for k in list(self.area.keys()):
            if self.area[k] == 0:
                del self.area[k]

    def energy_per_mac(self):
        return {k: v / self.macs for k, v in self.energy.items()}

    def energy_per_mac_1b(self):
        return {k: v / self.macs_1b for k, v in self.energy.items()}

    def per_mac(self, key: str):
        if key == "energy":
            return {k: v / self.macs for k, v in self.energy.items()}
        if key == "area":
            return {k: v / self.macs for k, v in self.area.items()}
        return getattr(self, key) / self.macs

    def per_mac_1b(self, key: str):
        if key == "energy":
            return {k: v / self.macs_1b for k, v in self.energy.items()}
        if key == "area":
            return {k: v / self.macs_1b for k, v in self.area.items()}
        return getattr(self, key) / self.macs_1b


class Comparison(dict):
    def __truediv__(self, other):
        return {k: v / other for k, v in self.items()}

    def __mul__(self, other):
        return {k: v * other for k, v in self.items()}

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)


class TestOutputList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name2test = {t.name: t for t in self}

    def consolidate_area_energy(self, from_keys: List[str], to: str):
        for t in self:
            t.consolidate_area_energy(from_keys, to)

    def consolidate_area(self, from_keys: List[str], to: str):
        for t in self:
            t.consolidate_area(from_keys, to)

    def consolidate_energy(self, from_keys: List[str], to: str):
        for t in self:
            t.consolidate_energy(from_keys, to)

    def assert_len_matches(self, reference_values: List[Any]):
        assert len(reference_values) == len(self), (
            f"Length of reference values ({len(reference_values)}) "
            f"does not match length of test outputs ({len(self)})"
        )

    def add_compare_ref(self, name: str, reference_values: List[Any]):
        self.assert_len_matches(reference_values)
        for t, v in zip(self, reference_values):
            t.add_compare_ref(name, v)

    def add_compare_ref_area(self, name: str, reference_values: List[Any]):
        if not isinstance(reference_values, Iterable):
            reference_values = [reference_values]
        self.assert_len_matches(reference_values)
        for t, v in zip(self, reference_values):
            t.add_compare_ref_area(name, v)

    def add_compare_ref_energy(self, name: str, reference_values: List[Any]):
        if not isinstance(reference_values, Iterable):
            reference_values = [reference_values]
        self.assert_len_matches(reference_values)
        for t, v in zip(self, reference_values):
            t.add_compare_ref_energy(name, v)

    def get_compare_ref_area(self):
        return [t.get_compare_ref_area() for t in self]

    def get_compare_ref_energy(self):
        return [t.get_compare_ref_energy() for t in self]

    def aggregate(self):
        return TestOutput.aggregate(self)

    def aggregate_by(self, *keys: str):
        return TestOutputList(TestOutput.aggregate_by(self, *keys))

    def split_by(self, *keys: str):
        to_agg = {}
        for t in self:
            key = tuple(t.access(k) for k in keys)
            to_agg[key] = to_agg.get(key, []) + [t]

        return [TestOutputList(v) for v in to_agg.values()]

    def clear_zero_energies(self):
        for t in self:
            t.clear_zero_energies()

    def clear_zero_areas(self):
        for t in self:
            t.clear_zero_areas()


def parse_timeloop_output(
    spec: tl.Specification,
    name: str,
    stats_path: str,
    art_path: str,
    accelergy_verbose: bool = False,
) -> TestOutput:
    cycles, computes, utilization, energy = parse_stats_file(stats_path)
    art_func = get_area_from_art_verbose if accelergy_verbose else get_area_from_art
    area = art_func(art_path)

    spec.parse_expressions()
    mapping = None
    if os.path.exists(stats_path.replace(".stats.txt", ".map.txt")):
        mapping = open(stats_path.replace(".stats.txt", ".map.txt")).read()

    cycle_seconds = spec.variables["GLOBAL_CYCLE_SECONDS"]

    return TestOutput(
        name,
        utilization,
        computes,
        cycles,
        cycle_seconds,
        energy,
        area,
        spec.variables,
        mapping,
    )
