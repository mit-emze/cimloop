from numbers import Number
from typing import Optional, List
from accelergy.plug_in_interface.estimator import Estimator, actionDynamicEnergy
from utils.bit_functions import *


class Capacitor(Estimator):
    name = "capacitor"
    percent_accuracy_0_to_100 = 80

    def __init__(
        self,
        capacitance: Number,
        technology: int,
        voltage: Number = 0.7,
        stacked: bool = False,
        cap_f_per_m2: Optional[Number] = None,
        border_area_m2: Optional[Number] = None,  # 1um^2 border
        energy_scale: Number = 1.0,
    ):
        super().__init__()
        self.capacitance = capacitance
        self.voltage = voltage
        self.stacked = stacked
        self.cap_f_per_m2 = (
            cap_f_per_m2 or 2.3e-3 * (22 / technology) ** 2
        )  # 2.3fF/um^2 @ 22nm
        self.border_area_m2 = (
            border_area_m2 or 1e-12 * (technology / 22) ** 2
        )  # 1um^2 @ 22nm
        self.energy_scale = energy_scale

    @actionDynamicEnergy
    def raise_voltage_to_from_non_supply(
        self, target_voltage: float, non_supply_voltage: float
    ) -> float:
        return self.capacitance * target_voltage * self.voltage

    @actionDynamicEnergy
    def raise_voltage_to(
        self, target_voltage: float, supply_voltage: float = None
    ) -> float:
        supply_voltage = self.voltage if supply_voltage is None else supply_voltage
        return self.capacitance * target_voltage * supply_voltage

    @actionDynamicEnergy
    def switch(
        self,
        value_probabilities: List[Number],
        zero_between_values: bool = True,
        supply_voltage: float = None,
    ) -> float:
        supply_voltage = self.voltage if supply_voltage is None else supply_voltage
        expected_energy = 0
        value_probabilities = rescale_sum_to_1(value_probabilities)
        for v0, p0 in enumerate(value_probabilities):
            for v1, p1 in enumerate(value_probabilities):
                v0 = 0 if zero_between_values else v0
                if v1 < v0:
                    continue
                e0 = self.raise_voltage_to(
                    v0 / (len(value_probabilities) - 1) * self.voltage, supply_voltage
                )
                e1 = self.raise_voltage_to(
                    v1 / (len(value_probabilities) - 1) * self.voltage, supply_voltage
                )
                expected_energy += (e1 - e0) * p0 * p1
        return expected_energy

    def get_charging_charge(
        self, value_probabilities: List[Number], charge_probability: float
    ) -> float:
        delta_v_avg = sum(v * p for v, p in enumerate(value_probabilities)) / (
            len(value_probabilities) - 1
        )
        return self.capacitance * self.voltage * delta_v_avg * charge_probability

    @actionDynamicEnergy
    def charge(
        self, value_probabilities: List[Number], charge_probability: Number = 0.0
    ) -> float:
        expected_energy = 0
        for v, p in enumerate(value_probabilities):
            v = v / (len(value_probabilities) - 1) * self.voltage
            prob = p * charge_probability
            expected_energy += self.raise_voltage_to(v) * prob
        return expected_energy

    def get_area(self) -> float:
        if self.stacked:  # Assume stacked on top of other components
            return 0
        return self.capacitance / self.cap_f_per_m2 + self.border_area_m2

    @actionDynamicEnergy
    def read(self):
        return self.raise_voltage_to(self.voltage) * self.energy_scale

    @actionDynamicEnergy
    def write(self):
        return 0

    @actionDynamicEnergy
    def update(self):
        return 0

    @actionDynamicEnergy
    def leak(self, global_cycle_seconds: float):
        return 0


class Wire(Capacitor):
    name = "wire"
    # 2e-10 from NeuroSim

    def __init__(
        self,
        length: Number,
        capacitance_per_m: Number = 2e-10,
        voltage: Number = 0.7,
        **kwargs,
    ):
        super().__init__(length * capacitance_per_m, voltage, **kwargs)
        self.length = length
        self.capacitance_per_m = capacitance_per_m
        self.voltage = voltage

    @actionDynamicEnergy
    def read(self):
        return super().read()

    @actionDynamicEnergy
    def write(self):
        return super().write()

    @actionDynamicEnergy
    def update(self):
        return super().update()

    @actionDynamicEnergy
    def leak(self, global_cycle_seconds: float):
        return 0

    def get_area(self):
        return 0


class PassGate(Estimator):
    """A basic D-Flip-Flop modeled using NeuroSim."""

    name = "pass_gate"
    percent_accuracy_0_to_100 = 50

    def __init__(self, technology: int, transistor_area_f2: int = 100):
        super().__init__()
        self.tech_node_m = technology * 1e-9
        self.transistor_area_f2 = transistor_area_f2

    def get_energy(self) -> float:
        return 0

    def get_area(self):
        n_transistors = 2
        return n_transistors * self.transistor_area_f2 * self.tech_node_m**2

    def leak(self, global_cycle_seconds: float):
        return 0
