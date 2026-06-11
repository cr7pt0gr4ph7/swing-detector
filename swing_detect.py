#!/usr/bin/env python3

import argparse
import json
import statistics
import sys

import numpy as np

from madmom.features.beats import RNNBeatProcessor
from madmom.features.beats import DBNBeatTrackingProcessor

from madmom.features.onsets import RNNOnsetProcessor
from madmom.features.onsets import OnsetPeakPickingProcessor


def detect_beats(audio_file):
    """
    Return beat times in seconds.
    """

    beat_activation = RNNBeatProcessor()(audio_file)

    beats = DBNBeatTrackingProcessor(
        fps=100,
        min_bpm=50,
        max_bpm=220,
    )(beat_activation)

    return np.asarray(beats)


def detect_onsets(audio_file):
    """
    Return onset times in seconds.
    """

    onset_activation = RNNOnsetProcessor()(audio_file)

    onsets = OnsetPeakPickingProcessor(
        fps=100,
        threshold=0.5,
        combine=0.03,
        pre_avg=0.1,
        post_avg=0.1,
        pre_max=0.02,
        post_max=0.02,
    )(onset_activation)

    return np.asarray(onsets)


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


def analyze_file(audio_file):
    beats = detect_beats(audio_file)

    if len(beats) < 8:
        raise RuntimeError(
            "Beat tracker found too few beats."
        )

    onsets = detect_onsets(audio_file)

    swing, stability, count = compute_swing(
        beats,
        onsets,
    )

    return {
        "file": audio_file,
        "swing": round(swing, 4),
        "stability": round(stability, 4),
        "intervals_used": count,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "audio_file",
        help="Audio file to analyze"
    )

    args = parser.parse_args()

    try:
        result = analyze_file(args.audio_file)

        print(
            json.dumps(
                result,
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
        sys.exit(1)


if __name__ == "__main__":
    main()
