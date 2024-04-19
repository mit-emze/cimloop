import math
from accelergy.plug_in_interface.estimator import Estimator, actionDynamicEnergy
from utils.bit_functions import *
from utils.bit_functions import *
from misc import *
from typing import List


def value2bits(value: int, resolution: int) -> List[int]:
    """Converts a value to a list of bits."""
    return [int(i) for i in bin(value)[2:].zfill(resolution)]


# Models of R-2R ladder and C-2C ladder DACs as described in the paper: A Charge
# Domain SRAM Compute-in-Memory Macro With C-2C Ladder-Based 8b MAC Unit in
# 22-nm FinFET Process for Edge Inference
#
# Wang, Hechen and Liu, Renzhi and Dorrance, Richard and Dasalukunte, Deepak and
# Lake, Dan and Carlton, Brent
#
# 10.1109/JSSC.2022.3232601


class DigitalAnalogConverterX2XLadder(Estimator):
    """X2X Ladder DAC Accelergy plug-in."""

    name = "dac_x2x_ladder"
    percent_accuracy_0_to_100 = 0

    def __init__(
        self,
        resolution: int,
        voltage: float,
        unit_x: float,
        technology: int,
        hist: List[float] = None,
        controller_energy_scale: float = 1,
    ):
        super().__init__()
        self.resolution = resolution
        self.voltage = voltage
        self.unit_x = unit_x
        self.technology = technology
        self.max_value = 2**resolution - 1
        self.hist = hist
        self.output_resistance = 0
        self.output_capacitance = 0
        self.name = f"{resolution}_bit_X2X_ladder_DAC"
        self.controller_energy_scale = controller_energy_scale

    def get_latency(
        self,
        load_cap: float,
        load_res: float,
        lsbs_expected_to_change: float = None,
        lsbs_allowed_incorrect: float = 0,
        porp_charge_loss_to_overcome: float = 0,
    ) -> float:
        """
        Args:
            load_cap (float): Load capacitance in Farads
            load_res (float): Load resistance in Ohms
            # LSBs changing between subsequent
            lsbs_expected_to_change (float, optional): Expected
            values. Lower values mean a lower-swing output. Defaults to None.
            lsbs_allowed_incorrect (float, optional): Number of LSBs allowed to be incorrect in
            tge final converted value. Defaults to 0.
            porp_charge_loss_to_overcome (float, optional): Porportion (0 to 1) of the charge on the
            load capacitor that is lost between subsequent cycles & must be recharged.
            Defaults to 0.
        Returns:
            float: Latency in seconds
        """
        if lsbs_expected_to_change is None:
            lsbs_expected_to_change = self.resolution

        lsbs_allowed_incorrect += 0.5

        # Time it takes for the output to converge within lsb_error_allowed LSB
        r = self.output_resistance + load_res
        c = self.output_capacitance + load_cap

        t0 = (
            -math.log(
                2**lsbs_expected_to_change / self.max_value
                + porp_charge_loss_to_overcome
            )
            * r
            * c
        )
        t1 = -math.log(2**lsbs_allowed_incorrect / self.max_value) * r * c
        return t1 - t0

    def solve_for_voltage_at_each_node(self, input_value: int) -> list:
        """
        Solves the matrix:
                         [ 4 -2  0....         ][V_0]
        input_voltages = [-2  5 -2  0 .....    ][V_1]
                         [ 0 -2  5 -2  0 ....  ][V_2]
                         [ 0  0 -2  5 -2 0 ....][V_3]
                         [ ................... ][V_4]
                         [ 0  0  0  0  0  -2  3][V_{n-1}]
        This matrix arises in C-2C and R-2R ladders. To solve, we set up the equation for each
        node: 0 = (V_{i} - V_{i-1}) + (V_{i} - V_{i+1}) + (V_{i} - Input_{i}) / 2
        The first node has an additional connector, and the last node does not
        Args:
            input_value (int): Input value to be converted. Must be between 0 and 2^resolution - 1.
        Returns:
            list: The voltage at each node in the ladder. The first element is the voltage at the
            voltage farthest from the output, and the last element is the voltage immediately
            before the output.
        """
        # Reverse input_value_bits to get the MSB on the right side of the circuit, closest to the
        # output
        input_value_bits = value2bits(input_value, self.resolution)[::-1]

        lhs = [i * self.voltage for i in input_value_bits]
        matrix_values = [[0, 4, -2]]
        for i in range(len(lhs) - 2):
            matrix_values.append([-2, 5, -2])
        matrix_values.append([-2, 3, 0])

        for i in range(len(lhs) - 2, -1, -1):
            mult = matrix_values[i][2] / matrix_values[i + 1][1]
            matrix_values[i][1] -= matrix_values[i + 1][0] * mult
            matrix_values[i][2] -= matrix_values[i + 1][1] * mult
            lhs[i] -= lhs[i + 1] * mult
            lhs[i] /= matrix_values[i][1]
            matrix_values[i] = [j / matrix_values[i][1] for j in matrix_values[i]]

        for i in range(0, len(lhs) - 1):
            mult = matrix_values[i + 1][0] / matrix_values[i][1]
            matrix_values[i + 1][0] -= matrix_values[i][1] * mult
            matrix_values[i + 1][1] -= matrix_values[i][2] * mult
            lhs[i + 1] -= lhs[i] * mult

        lhs = [l / matrix_values[i][1] for i, l in enumerate(lhs)]

        # Un-reverse the bits
        return lhs[::-1]

    def input_value_to_analog_energy(self, input_value: int) -> float:
        """Returns the energy in Joules to convert the input value to an analog voltage"""
        input_value_bits = value2bits(input_value, self.resolution)
        node_voltages = self.solve_for_voltage_at_each_node(input_value)
        current = 0
        for i, bit in enumerate(input_value_bits):
            current += (self.voltage - node_voltages[i]) * (bit != 0)
        energy = current * self.voltage * self.unit_x
        return energy

    def convert_value(
        self, input_value: int, latency: float = 1e-9, load_cap: float = 10e-15
    ) -> float:
        """Returns the energy in Joules to convert the input value to an analog voltage"""
        # Ignore unused parameters
        _ = latency, load_cap
        return self.input_value_to_analog_energy(input_value)

    @actionDynamicEnergy
    def convert(
        self,
        latency: float = 1e-9,
        load_cap: float = 10e-15,
        ignore_controller_energy: bool = False,
    ):
        """Returns the average energy in Joules to convert the input value to an analog voltage"""
        energy = 0
        prob = 0

        # This code resizes the histogram into the full distribution of values
        # that this DAC can produce. It also makes sure to map the 0
        # probability exactly.
        newhist = [0] * 2**self.resolution
        idx0 = len(self.hist) // 2
        new_idx0 = len(newhist) // 2
        width_scale = len(self.hist) / len(newhist)
        prunedhist = [i for i in self.hist]
        for exact_maps in [(0, 0), (idx0, new_idx0)]:
            newhist[exact_maps[1]] = prunedhist[exact_maps[0]] / min(width_scale, 1)
            prunedhist[exact_maps[0]] = 0

        for i in range(len(newhist)):
            if i == new_idx0 or i == 0:
                continue
            loc = i / len(newhist) * (len(prunedhist) - 1)
            if width_scale > 1:
                start = math.floor(loc)
                end = min(math.ceil(start + width_scale), len(prunedhist) - 1)
                for j in range(start, end):
                    newhist[i] += prunedhist[j]
            else:
                porp = loc - math.floor(loc)
                newhist[i] += prunedhist[math.floor(loc)] * (1 - porp)
                newhist[i] += prunedhist[math.ceil(loc)] * porp

        # Calculate the energy
        for i, p in enumerate(newhist):
            energy += self.convert_value(i, latency, load_cap) * p

        energy /= sum(newhist)
        if ignore_controller_energy:
            return energy
        return energy + self.get_controller_energy()

    @actionDynamicEnergy
    def leak(self, global_cycle_seconds: float):
        return 0

    def get_controller_energy(self):
        # 0.08pJ/bit at 32nm 1.0V
        return (
            (self.voltage * self.technology) ** 2
            * self.resolution
            * 8e-17
            * self.controller_energy_scale
        )

    def get_area(self):
        raise NotImplementedError(
            "X2X DAC should not be instantiated directly. Use a subclass."
        )


class DigitalAnalogConverter_C2C(DigitalAnalogConverterX2XLadder):
    """C-2C Ladder DAC."""

    name = "dac_c2c_ladder"
    percent_accuracy_0_to_100 = 80

    def __init__(
        self,
        resolution: int,
        voltage: float,
        unit_capacitance: float,
        technology: int,
        hist: List[float] = None,
        capacitors_are_stacked: bool = False,
    ):
        super().__init__(
            resolution=resolution,
            voltage=voltage,
            unit_x=unit_capacitance,
            technology=technology,
            hist=hist,
        )
        self.unit_capacitance = unit_capacitance
        self.output_capacitance += unit_capacitance * 2
        self.name = f"{resolution}_bit_C2C_ladder_DAC"
        self.cap = Capacitor(
            self.voltage, capacitance=unit_capacitance, stacked=capacitors_are_stacked
        )

    def get_area(self):
        return self.cap.get_area() * self.resolution


class DigitalAnalogConverter_R2R(DigitalAnalogConverterX2XLadder):
    """R-2R ladder DAC."""

    name = "dac_r2r_ladder"
    percent_accuracy_0_to_100 = 80

    def __init__(
        self,
        resolution: int,
        voltage: float,
        unit_resistance: float,
        technology: int,
        hist: List[float] = None,
        area_scale: float = 1,
    ):
        super().__init__(
            resolution=resolution,
            voltage=voltage,
            unit_x=1 / unit_resistance / 2,
            technology=technology,
            hist=hist,
        )
        self.unit_resistance = unit_resistance
        self.output_resistance += unit_resistance
        self.name = f"{resolution}_bit_R2R_ladder_DAC"
        self.m2_chip_area_per_ohm = 0.7e-14 * (technology / 22) ** 2 * area_scale

    @actionDynamicEnergy
    def convert(
        self,
        action_latency_cycles: float,
        cycle_seconds: float,
        load_cap: float = 10e-15,
    ):
        """Returns the average energy in Joules to convert the input value to an analog voltage"""
        min_latency = self.get_latency(load_cap=load_cap, load_res=0)
        latency = action_latency_cycles * cycle_seconds
        if latency < min_latency:
            self.logger.warning(
                f"Latency {latency} is less than the DAC latency {min_latency}."
                f"DAC output may have errors due to insufficient time to settle."
            )
        power = super().convert(latency, load_cap, ignore_controller_energy=True)
        controller_energy = self.get_controller_energy()
        self.logger.info(
            f"R2R DAC consumes {power}W for {latency} seconds. "
            f"Controller energy: {controller_energy}J"
        )
        return power * latency + controller_energy

    def get_area(self):
        # 1.4e-14 ohms/um^2 chip area
        return self.m2_chip_area_per_ohm * self.resolution * self.unit_resistance


# Generate a 5kohm unit resistance R-2R ladder with 8b resolution
# Print, for each input value, the energy to convert to an analog voltage
# over 5ns
if __name__ == "__main__":
    dac = DigitalAnalogConverter_R2R(
        resolution=8,
        voltage=0.7,
        unit_resistance=5000,
        technology=22,
    )
    avg = 0
    for i in range(2**8):
        print(f"{i}: {dac.convert_value(i, latency=5e-9)}")
        avg += dac.convert_value(i, latency=5e-9) * 1e6

    print(f"Avg: {avg / 2 ** 8}")
