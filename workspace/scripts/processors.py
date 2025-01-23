import pytimeloop.timeloopfe.v4 as tl


class ArrayContainer(tl.arch.Container):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        tl.arch.ArchNodes.add_attr("!ArrayContainer", ArrayContainer)


class MaxUtilizationDescriptorTop(tl.DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("spatial", MaxUtilizationDescriptor, None)
        super().add_attr("temporal", MaxUtilizationDescriptor, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial: MaxUtilizationDescriptor = self["spatial"]
        self.temporal: MaxUtilizationDescriptor = self["temporal"]


class MaxUtilizationDescriptor(tl.DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("factors", tl.constraints.Factors)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factors: tl.constraints.Factors = self["factors"]


class ArrayProcessor(tl.processors.Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def declare_attrs(self, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr(tl.problem.Problem, "name", str, None)
        super().add_attr(tl.problem.Problem, "dnn_name", str, None)
        super().add_attr(tl.problem.Problem, "notes", str, None)
        super().add_attr(tl.problem.Problem, "histograms", dict, {})
        super().add_attr(
            tl.arch.Leaf, "max_utilization", MaxUtilizationDescriptorTop, None
        )
        MaxUtilizationDescriptorTop.declare_attrs()
        MaxUtilizationDescriptor.declare_attrs()
        ArrayContainer.declare_attrs()

    def fetch_integer(self, spec: tl.Specification, node: tl.Node, key: str):
        v = node[key]
        if v in spec.variables:
            v = spec.variables[v]

        errstr = f"Non-integer value {v} for {key} in {node}"
        try:
            assert float(v).is_integer(), errstr
            return int(v)
        except ValueError as e:
            raise ValueError(errstr) from e

    def expand_utilization(self, spec: tl.Specification):
        expanded = {"C": 1, "M": 1}
        instance = spec.problem.instance
        for l in spec.get_nodes_of_type(tl.arch.Leaf):
            if l.constraints.spatial.get("factors_only", None) is not None:
                continue
            if l.max_utilization is not None:
                continue
            f = (
                l.spatial.get_fanout()
                // l.constraints.spatial.factors.get_minimum_product(instance)
            )
            if f <= 1:
                continue

            remaining_multipliers = []
            for m in expanded:
                if any(
                    m in spec.problem.shape.dataspace2dims(d)
                    for d in (l.constraints.spatial.no_iteration_over_dataspaces or [])
                ):
                    continue
                remaining_multipliers.append(m)

            mult_warning = {}
            prev_instance = dict(instance)

            if remaining_multipliers:
                for f in num2list_of_prime_factors(f):
                    target = min(remaining_multipliers, key=lambda x: instance[x])
                    prev = instance[target]
                    mult_warning[target] = f * mult_warning.get(target, 1)
                    instance[target] *= f

            if mult_warning:
                k = f"{','.join(mult_warning)}"
                t = f"({','.join(str(m) for m in mult_warning.values())})"
                s = f"({','.join(str(prev_instance[m]) for m in mult_warning)})"
                e = f"({','.join(str(instance[m]) for m in mult_warning)})"
                self.logger.warning(
                    f"To fill up {l.name}, multiplied {k} by {t}: {s} -> {e}"
                )
        return expanded

    def pre_parse_process(self, spec: tl.Specification):
        super().pre_parse_process(spec)
        # Pop the relevant items from the problem
        for x in ["name", "dnn_name", "notes"]:
            spec.problem.pop(x)

        histograms = spec.problem.pop("histograms")
        for k, v in histograms.items():
            spec.variables[f"{k.upper()}_HIST"] = v
        for k in list(spec.variables.keys()):
            if not k.endswith("_HIST"):
                spec.variables[k] = spec.variables.pop(k)

        prob = spec.problem
        n_rows = 1
        n_cols = 1

        parallel_dataspaces = {ds.name: 1 for ds in prob.shape.data_spaces}

        for cim_container in spec.get_nodes_of_type(ArrayContainer):
            constraints = cim_container.constraints
            cim_container.attributes["_is_CiM"] = True

            spatial = cim_container.spatial
            meshX = self.fetch_integer(spec, spatial, "meshX")
            meshY = self.fetch_integer(spec, spatial, "meshY")
            assert meshX == 1 or meshY == 1, (
                f"Either meshX or meshY must be 1 in {spatial}. Got "
                f"{meshX=} and {meshY=}."
            )
            n_rows *= meshY
            n_cols *= meshX
            constraints.spatial.split = 99999 if meshY == 1 else 0
            for ds in parallel_dataspaces:
                if ds in (constraints.spatial.no_reuse or []):
                    parallel_dataspaces[ds] *= meshX * meshY

        v = spec.variables
        v["ARRAY_WORDLINES"] = f'{n_rows} * ({v["CIM_UNIT_DEPTH_CELLS"]})'
        v["ARRAY_BITLINES"] = f'{n_cols} * ({v["CIM_UNIT_WIDTH_CELLS"]})'

        for ds, n in parallel_dataspaces.items():
            v[f"ARRAY_PARALLEL_{ds.upper()}"] = n

        # Move ARRAY_WORDLINES and ARRAY_BITLINES to the top of the list
        for k in list(v.keys()):
            if k not in ["ARRAY_WORDLINES", "ARRAY_BITLINES"]:
                # If it can be casted directly to a number, it doesn't need to
                # be moved to the bottom.
                try:
                    float(v[k])
                except (ValueError, TypeError):
                    v[k] = v.pop(k)

    def process(self, spec: tl.Specification):
        if not spec.variables.pop("MAX_UTILIZATION", False):
            for l in spec.get_nodes_of_type(tl.arch.Leaf):
                l.pop("max_utilization", None)
            return

        max_util_shape = {
            "X": spec.variables["ENCODED_INPUT_BITS"],
            "Y": spec.variables["ENCODED_WEIGHT_BITS"],
            "Z": spec.variables["ENCODED_OUTPUT_BITS"],
        }
        for leaf in spec.get_nodes_of_type(tl.arch.Leaf):
            if (mu := leaf.max_utilization) is None:
                continue

            for target in ["spatial", "temporal"]:
                if (t := getattr(mu, target)) is None:
                    continue
                for k, eq, v in t.factors.get_split_factors():
                    assert eq == "=", (
                        f"Only '=' is supported for maximum utilization factors. "
                        f"Got {eq} in {t.factors}."
                    )
                    max_util_shape[k] = max_util_shape.get(k, 1) * v
                    # leaf.constraints[target].factors.add_eq_factor(k, v, overwrite=True)

        max_util_shape.setdefault("N", 1)

        instance = spec.problem.instance
        instance.update(max_util_shape)
        for k, e in self.expand_utilization(spec).items():
            instance[k] = instance.get(k, 1) * e
        instance["M"] *= spec.variables["CIM_UNIT_DEPTH_CELLS"]

        weight_slice_spill = spec.variables["N_WEIGHT_SLICES"]
        assert instance["M"] >= weight_slice_spill, (
            f"To map this problem, {weight_slice_spill} weight slices must "
            f"be mapped. We could map these to parallel output channels M, but "
            f"there are only {instance['M']} available."
        )
        instance["M"] //= weight_slice_spill

        for l in spec.get_nodes_of_type(tl.arch.Leaf):
            l.pop("max_utilization", None)


def num2list_of_prime_factors(x: int):
    factors = []
    while x > 1:
        for i in range(2, x + 1):
            if x % i == 0:
                factors.append(i)
                x //= i
                break
    return factors
