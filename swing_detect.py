#!/usr/bin/env python3

import argparse
import json
import statistics
import sys

from dataclasses import dataclass
from dataclasses_json import dataclass_json

import numpy as np

import librosa


def compute_swing(beats, onsets):
    """
    Estimate swing ratio from beat intervals.

    Returns:
        swing_score
        stability
        interval_count
    """

    swing_values = []

    for i in range(len(beats) - 1):
        b0 = beats[i]
        b1 = beats[i + 1]

        duration = b1 - b0

        if duration <= 0:
            continue

        midpoint = b0 + duration * 0.5

        candidates = onsets[
            (onsets > b0 + duration * 0.15)
            & (onsets < b1 - duration * 0.15)
        ]

        if len(candidates) == 0:
            continue

        distances = np.abs(candidates - midpoint)

        idx = np.argmin(distances)

        onset = candidates[idx]

        normalized_position = (onset - b0) / duration

        # Reject unlikely values.
        if normalized_position < 0.20:
            continue

        if normalized_position > 0.90:
            continue

        # Reject intervals where several onsets compete.
        competing = np.sum(
            np.abs(candidates - midpoint)
            < duration * 0.10
        )

        if competing > 3:
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

    return swing, stability, len(swing_values)


@dataclass_json
@dataclass
class AudioFeatures:
    file: str
    bpm: float
    samplerate: float
    swing: float
    swing_stability: float
    swing_intervals_used: float


def analyze_file(audio_file, offset: float | None = None, duration: float | None = None):
    data, rate = librosa.load(audio_file, offset=offset, duration=duration)
    tempo, beats = librosa.beat.beat_track(y=data, sr=rate, units='time')
    onsets = librosa.onset.onset_detect(y=data, sr=rate, units='time')

    swing, stability, count = compute_swing(beats, onsets)

    return AudioFeatures(
        file=audio_file if isinstance(audio_file, str) else audio_file.name if hasattr(
            audio_file, "name") else None,
        bpm=tempo,
        samplerate=rate,
        swing=round(swing, 4),
        swing_stability=round(stability, 4),
        swing_intervals_used=count,
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
