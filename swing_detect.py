#!/usr/bin/env python3

import argparse
import json
import statistics
import sys

from dataclasses import dataclass
from dataclasses_json import dataclass_json

import numpy as np
import numpy.typing as npt

import librosa


def compute_swing(
    beats: npt.NDArray[np.float32],
    onsets: npt.NDArray[np.float32],
    min_position: float = 0.4,
    max_position: float = 0.8,
    max_competing: int = 1,
):
    """
    Estimate swing ratio from beat intervals.

    Returns:
        swing_amount
        swing_stability
        interval_count
    """

    swing_values = []

    for i in range(len(beats) - 1):
        b0: float = beats[i]
        b1: float = beats[i + 1]

        duration = b1 - b0

        if duration <= 0:
            continue

        midpoint = b0 + duration * 0.5

        # Find onsets for the current beat section. Reject unlikely
        # onset events that are too close to the main beats.
        candidates = onsets[
            (onsets > b0 + duration * min_position)
            & (onsets < b1 - duration * (1.0 - max_position))
        ]

        if len(candidates) == 0:
            continue

        # Select onset event closest to the midpoint
        distances = np.abs(candidates - midpoint)
        idx = np.argmin(distances)
        onset: float = candidates[idx]
        normalized_position = (onset - b0) / duration

        # Reject intervals where several onsets are close to the midpoint
        competing = np.sum(
            np.abs(candidates - midpoint)
            < duration * 0.10
        )

        if competing > max_competing:
            continue

        swing_values.append(float(normalized_position))

    if len(swing_values) < 10:
        raise RuntimeError(
            "Too few valid intervals found."
        )

    swing = statistics.median(swing_values)

    stability = statistics.median(
        [abs(x - swing) for x in swing_values]
    )

    return swing, stability, swing_values, len(swing_values)


@dataclass_json
@dataclass
class AudioFeatures:
    file: str
    bpm: float
    samplerate: float
    swing: float
    swing_stability: float
    swing_intervals_used: float
    swing_values: list[float]
    swing_histogram: any


def analyze_file(
    audio_file,
    offset: float | None = None,
    duration: float | None = None,
    raw_swing_values: bool = False,
    raw_swing_histogram: bool = False,
):
    data, rate = librosa.load(audio_file, offset=offset, duration=duration)
    tempo, beats = librosa.beat.beat_track(y=data, sr=rate, units='time')
    onsets = librosa.onset.onset_detect(y=data, sr=rate, units='time')

    swing, stability, swing_values, count = compute_swing(beats, onsets)
    swing_histogram_counts, swing_histogram_edges = np.histogram(swing_values, 40, (0.4,0.8))
    swing_histogram = np.column_stack((swing_histogram_counts, swing_histogram_edges[0:-1]))

    print(swing_histogram)

    return AudioFeatures(
        file=audio_file if isinstance(audio_file, str) else audio_file.name if hasattr(
            audio_file, "name") else None,
        bpm=tempo,
        samplerate=rate,
        swing=round(swing, 4),
        swing_stability=round(stability, 4),
        swing_intervals_used=count,
        swing_values=swing_values if raw_swing_values else [],
        swing_histogram=swing_histogram.tolist() if raw_swing_histogram else [],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "audio_files",
        help="Audio file(s) to analyze",
        nargs="*",
    )

    args = parser.parse_args()
    has_error = False

    for audio_file in args.audio_files:
        try:
            result = analyze_file(audio_file)

            print(
                result.to_json(
                    indent=2,
                )
            )

        except Exception as e:
            print(
                json.dumps(
                    {"error": str(e)},
                    indent=2,
                ),
                file=sys.stderr,
            )
            has_error = True

    if has_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
