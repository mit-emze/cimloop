import os
from typing import Any, Iterable, Tuple, List, Union
import pytimeloop.timeloopfe.v4 as tl
from pytimeloop.timeloopfe.v4.output_parsing import MultipliableDict
import yaml


class MacroOutputStats(tl.output_parsing.OutputStats):
    def __init__(self, *args, scale_computes: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        v = self.variables
        self.input_bits = v["INPUT_BITS"]
        self.weight_bits = v["WEIGHT_BITS"]
        self.output_bits = v["OUTPUT_BITS"]
        self.encoded_input_bits = v["ENCODED_INPUT_BITS"]
        self.encoded_weight_bits = v["ENCODED_WEIGHT_BITS"]
        self.encoded_output_bits = v["ENCODED_OUTPUT_BITS"]

        # Undo the bitwise virtualization
        if scale_computes:
            self.scale_computes_by(
                1
                / (
                    self.encoded_input_bits
                    * self.encoded_weight_bits
                    * self.encoded_output_bits
                )
            )

        # Calculate one-bit equivalent stats
        n_1b = self.input_bits * self.weight_bits
        self.computes_1b = self.computes * n_1b
        self.computes_per_second_1b = self.computes_per_second * n_1b
        self.computes_per_joule_1b = self.computes_per_joule * n_1b

        self.tops = self.computes / (self.cycles * self.cycle_seconds) / 1e12 * 2
        self.tops_per_mm2 = self.tops / self.area / 1e6
        self.tops_per_w = self.computes / self.energy * 2 / 1e12

        self.tops_1b = self.tops * n_1b
        self.tops_per_mm2_1b = self.tops_per_mm2 * n_1b
        self.tops_per_w_1b = self.tops_per_w * n_1b

    @staticmethod
    def from_output_stats(
        output_stats: tl.output_parsing.OutputStats, scale_computes: bool = True
    ):
        return MacroOutputStats(
            output_stats.percent_utilization,
            output_stats.computes,
            output_stats.cycles,
            output_stats.cycle_seconds,
            output_stats.per_component_energy,
            output_stats.per_component_area,
            output_stats.variables,
            output_stats.mapping,
            scale_computes=scale_computes,
        )

    @staticmethod
    def aggregate(*args, **kwargs):
        return MacroOutputStats.from_output_stats(  # Don't re-scale
            tl.output_parsing.OutputStats.aggregate(*args, **kwargs),
            scale_computes=False,
        )

    @staticmethod
    def aggregate_by(*args, **kwargs):
        return MacroOutputStatsList(
            [
                MacroOutputStats.from_output_stats(t, scale_computes=False)
                for t in tl.output_parsing.OutputStats.aggregate_by(*args, **kwargs)
            ]
        )

    def add_compare_ref(self, name: str, reference_value: Any):
        setattr(
            self,
            name,
            MultipliableDict(reference=reference_value, model=getattr(self, name)),
        )

    def add_compare_ref_area(self, name: str, reference_value: Any):
        self.per_component_area[name] = MultipliableDict(
            reference=reference_value, model=self.per_component_area[name]
        )

    def add_compare_ref_energy(self, name: str, reference_value: Any):
        self.per_component_energy[name] = MultipliableDict(
            reference=reference_value, model=self.per_component_energy[name]
        )

    def _get_compare_ref(self, name: str):
        d = getattr(self, name)
        return MultipliableDict(
            **{k: v for k, v in d.items() if isinstance(v, MultipliableDict)}
        )

    def get_compare_ref_area(self):
        return self._get_compare_ref("per_component_area")

    def get_compare_ref_energy(self):
        return self._get_compare_ref("per_component_energy")


class MacroOutputStatsList(tl.output_parsing.OutputStatsList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        return MacroOutputStats.aggregate(self)

    def aggregate_by(self, *keys: str):
        return MacroOutputStatsList(MacroOutputStats.aggregate_by(self, *keys))

    def split_by(self, *keys: str):
        to_agg = {}
        for t in self:
            key = tuple(t.access(k) for k in keys)
            to_agg[key] = to_agg.get(key, []) + [t]

        return [MacroOutputStatsList(v) for v in to_agg.values()]

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
) -> MacroOutputStats:
    cycles, computes, utilization, energy = parse_stats_file(stats_path)
    art_func = get_area_from_art_verbose if accelergy_verbose else get_area_from_art
    area = art_func(art_path)

    spec.parse_expressions()
    mapping = None
    if os.path.exists(stats_path.replace(".stats.txt", ".map.txt")):
        mapping = open(stats_path.replace(".stats.txt", ".map.txt")).read()

    cycle_seconds = spec.variables["GLOBAL_CYCLE_SECONDS"]

    return MacroOutputStats(
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
