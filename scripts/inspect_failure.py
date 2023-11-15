#!/bin/env python3

from dataclasses import dataclass
from typing import List
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt


@dataclass(init=False)
class TestFailureData:
    input_signal: List[float]
    output_signal: List[float]
    analysis_peaks: List[int]
    synthesis_peaks: List[int]
    input_frequency: float
    target_frequency: float
    detected_output_frequency: float
    diff: float
    buffer_size: int
    sample_rate: int

    def construct_from_json(self, serialized_json: str):
        parsed_json = json.loads(serialized_json)
        self.input_signal = parsed_json["input"]
        self.output_signal = parsed_json["output"]
        self.analysis_peaks = parsed_json["analysis_peaks"]
        self.synthesis_peaks = parsed_json["synthesis_peaks"]
        self.input_frequency = parsed_json["input_frequency"]
        self.target_frequency = parsed_json["target_frequency"]
        self.detected_output_frequency = parsed_json[
            "detected_output_frequency"
        ]
        self.diff = parsed_json["diff"]
        self.buffer_size = parsed_json["buffer_size"]
        self.sample_rate = parsed_json["sample_rate"]


INPUT_COLOR = "#63e0ff"
OUTPUT_COLOR = "#a463ff"


def visualize(data: TestFailureData):
    sample_numbers = [i for i in range(len(data.input_signal))]

    plt.plot(
        sample_numbers,
        data.input_signal,
        color=INPUT_COLOR,
        label=f"Input ({data.input_frequency} Hz)",
    )
    plt.plot(
        sample_numbers,
        data.output_signal,
        color=OUTPUT_COLOR,
        label=f"Output ({data.detected_output_frequency} Hz)",
    )

    plt.plot(
        data.analysis_peaks,
        [data.input_signal[peak] for peak in data.analysis_peaks],
        color=INPUT_COLOR,
        label="Analysis Peaks",
        marker="o",
        markersize=10,
        linestyle="",
    )

    plt.plot(
        data.synthesis_peaks,
        [data.output_signal[peak] for peak in data.synthesis_peaks],
        color=OUTPUT_COLOR,
        label="Synthesis Peaks",
        marker="o",
        markersize=10,
        linestyle="",
    )

    plt.title(
        f"Shift {data.input_frequency} to {data.target_frequency} (Buffer Size {data.buffer_size} Sample Rate {data.sample_rate})"
    )
    plt.xlabel("Sample #")
    plt.ylabel("Amplitude")
    plt.legend(loc="lower left")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="inspect_failure.py", description="Inspect a failed test case."
    )
    parser.add_argument("file", help="The file containing failed test data")
    args = parser.parse_args()

    data = TestFailureData()
    data.construct_from_json(Path(args.file).read_text())

    visualize(data)
