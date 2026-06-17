#!/usr/bin/env python3

import argparse
import json
import statistics
import sys

from dataclasses import dataclass
from typing import Literal
from dataclasses_json import dataclass_json
from mutagen.easyid3 import EasyID3

import mutagen
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


def sum_energies(
    spectrum: npt.NDArray[np.floating],
    freqs: npt.NDArray[np.floating],
    low_freq: float,
    high_freq: float,
    normalize: bool = False,
):
    # Calculate the total energy of the given frequency band
    energy = np.sum(
        spectrum[
            (freqs >= low_freq) &
            (freqs < high_freq)
        ]
    )

    # Normalize for the band size
    normalized_energy = energy / (high_freq - low_freq)

    if normalize:
        return normalized_energy
    else:
        return energy


def sliding_mean_spectrum(
    raw_spectrum: npt.NDArray[np.floating],
    frame: int,
    pre: int = 0,
    post: int = 0,
):
    start = max(frame - pre, 0)
    end = min(frame + 1 + post, raw_spectrum.shape[1])
    return np.mean(raw_spectrum[:, start:end], axis=1)


def detect_kick_snare(audio_file):
    data, rate = librosa.load(audio_file, sr=None, mono=True)

    # Hop length (in samples)
    hop_length = 512

    # Length of windowed FFT signal after padding
    n_fft = 2048

    onset_frames = librosa.onset.onset_detect(
        y=data,
        sr=rate,
        units='frames',
        hop_length=hop_length,
        backtrack=True,
    )
    raw_spectrum: npt.NDArray[np.complexfloating] = np.abs(
        librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
    freqs: npt.NDArray[np.floating] = librosa.fft_frequencies(
        sr=rate, n_fft=n_fft)

    frames: list[float] = []
    ratios: list[float] = []
    sub_energies: list[float] = []
    low_mid_energies: list[float] = []
    presence_energies: list[float] = []
    high_energies: list[float] = []

    for frame in onset_frames:
        # Compute sliding mean over 4 frames
        spectrum = sliding_mean_spectrum(raw_spectrum, frame, pre=1, post=3)

        # Sum up energies over low / high frequency bands
        normalize = False  # Whether to normalize for the size of the frequency bands
        sub_energy = sum_energies(
            spectrum, freqs, 20, 120, normalize=normalize)
        low_mid_energy = sum_energies(
            spectrum, freqs, 120, 500, normalize=normalize)
        presence_energy = sum_energies(
            spectrum, freqs, 1000, 4000, normalize=normalize)
        high_energy = sum_energies(
            spectrum, freqs, 5000, 12000, normalize=normalize)

        frames.append(frame)
        ratios.append(low_mid_energy / (presence_energy + 1e-10))
        sub_energies.append(sub_energy)
        low_mid_energies.append(low_mid_energy)
        presence_energies.append(presence_energy)
        high_energies.append(high_energy)

    frame_times = librosa.frames_to_time(
        frames, sr=rate, hop_length=hop_length)

    ratio_times = np.column_stack(
        (frame_times, ratios, sub_energies,
         low_mid_energies, presence_energies, high_energies)).tolist()

    return ratio_times


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
    data, rate = librosa.load(
        audio_file,

        # Use native sample rate of audio file
        sr=None,

        # Start + Duration of segment to analyze
        offset=offset,
        duration=duration,
    )

    tempo, beats = librosa.beat.beat_track(y=data, sr=rate, units='time')
    onsets = librosa.onset.onset_detect(y=data, sr=rate, units='time')

    swing, stability, swing_values, count = compute_swing(beats, onsets)

    # Compute histogram of swing values
    swing_histogram_counts, swing_histogram_edges = np.histogram(
        swing_values, 40, (0.4, 0.8))
    swing_histogram = np.column_stack(
        (swing_histogram_counts, swing_histogram_edges[0:-1]))

    return AudioFeatures(
        file=audio_file if isinstance(audio_file, str) else audio_file.name if hasattr(
            audio_file, "name") else None,
        bpm=tempo,
        samplerate=rate,
        swing=round(swing, 4),
        swing_stability=round(stability, 4),
        swing_intervals_used=count,
        swing_values=swing_values if raw_swing_values else None,
        swing_histogram=swing_histogram.tolist() if raw_swing_histogram else None,
    )


class OutputFormat:
    def output(self, v: any):
        pass

    def error(self, e: Exception):
        pass


class QuietFormat(OutputFormat):
    pass


class JsonFormat(OutputFormat):
    def output(self, v: any):
        if hasattr(v, "to_json"):
            print(
                v.to_json(
                    indent=2,
                )
            )
        else:
            print(
                json.dumps(
                    v,
                    indent=2,
                )
            )

    def error(self, e: Exception):
        print(
            json.dumps(
                {"error": str(type(e)) + ": " + str(e)},
                indent=2,
            ),
            file=sys.stderr,
        )


def get_output_format(format: str | None) -> OutputFormat:
    if format == "json":
        return JsonFormat()
    elif format == "quiet":
        return QuietFormat()
    elif format is None:
        return JsonFormat()  # Default to JSON
    else:
        raise ValueError("Invalid output format: " + format)


def write_onsets(audio_file, units: Literal['frames', 'samples', 'time']):
    data, rate = librosa.load(audio_file, sr=None)
    tempo, beats = librosa.beat.beat_track(y=data, sr=rate, units=units)
    onsets = librosa.onset.onset_detect(y=data, sr=rate, units=units)

    # Write event times to CSV files
    with open(audio_file + ".beats_" + units + ".csv", "wt") as beats_file:
        beats_file.writelines([str(time) + "\n" for time in beats])

    with open(audio_file + ".onsets_" + units + ".csv", "wt") as onsets_file:
        onsets_file.writelines([str(time) + "\n" for time in onsets])


def write_onset_strengths(audio_file):
    data, rate = librosa.load(audio_file)
    onset_strengths = librosa.onset.onset_strength(y=data, sr=rate)

    with open(audio_file + ".onset_strengths.csv", "wt") as onset_strengths_file:
        onset_strengths_file.writelines(
            [str(strength) + "\n" for strength in onset_strengths])


def write_onset_types(audio_file):
    ratio_times = detect_kick_snare(audio_file)

    with open(audio_file + ".onset_types.csv", "wt") as onset_types_file:
        onset_types_file.write(
            "Time,Ratio,Sub Energy,Low-Mid Energy,Presence Energy,High Energy\n")
        onset_types_file.writelines(
            [",".join([str(field) for field in data]) + "\n" for data in ratio_times])


def cmd_write_onsets_for_audio_file(audio_file, output_format: OutputFormat, args: argparse.Namespace):
    write_onset_strengths(audio_file)
    write_onset_types(audio_file)
    write_onsets(audio_file, 'time')
    write_onsets(audio_file, 'frames')
    write_onsets(audio_file, 'samples')


def cmd_analyze_audio_file(audio_file, output_format: OutputFormat, args: argparse.Namespace):
    result = analyze_file(
        audio_file,
        offset=args.offset,
        duration=args.duration,
    )
    output_format.output(result)

    # Write tags to audio files
    if args.write_tags:
        metadata: mutagen.FileType = mutagen.File(
            audio_file, easy=True)

        if metadata is None:
            raise NotImplementedError(
                "Failed to get metadata from audio file: " + audio_file)

        if args.swing_amount_tag and (args.overwrite_tags or args.swing_amount_tag not in metadata.tags):
            metadata.tags[args.swing_amount_tag] = [str(result.swing)]

        if args.swing_stability_tag and (args.overwrite_tags or args.swing_stability_tag not in metadata.tags):
            metadata.tags[args.swing_stability_tag] = [
                str(result.swing_stability)]

        print(metadata.tags.pprint())

        metadata.save()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--offset",
        help="Start analysis at this time within the audio file (in seconds).",
        type=float,
    )

    parser.add_argument(
        "-d", "--duration",
        help="Only analyze up to this much audio (in seconds).",
        type=float,
    )

    parser.add_argument(
        "-f", "--format",
        help="Output format",
        type=str,
        choices=["json", "quiet"],
        default="json",
    )

    parser.add_argument(
        "-w", "--write-tags",
        help="Write the detected swing amount to the ID3 tags of the audio files.",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--no-overwrite-tags",
        dest="overwrite_tags",
        help="Do not overwrite existing ID3 tags.",
        action='store_false',
    )

    parser.add_argument(
        "--swing-amount-tag",
        help="Name of the metadata tag to write the detected swing amount to.",
        type=str,
        default="swing_amount",
    )

    parser.add_argument(
        "--swing-stability-tag",
        help="Name of the metadata tag to write the detected swing amount to.",
        type=str,
        # default="swing_stability",
    )

    parser.add_argument(
        "--write-onsets",
        help="Write CSV files with the beats, note onsets etc.",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "audio_files",
        help="Audio file(s) to analyze",
        nargs="+",
    )

    args = parser.parse_args()
    output_format = get_output_format(args.format)
    has_error = False

    if args.swing_amount_tag:
        EasyID3.RegisterTXXXKey(args.swing_amount_tag, args.swing_amount_tag)

    if args.swing_stability_tag:
        EasyID3.RegisterTXXXKey(args.swing_stability_tag,
                                args.swing_stability_tag)

    for audio_file in args.audio_files:
        try:
            if args.write_onsets:
                cmd_write_onsets_for_audio_file(
                    audio_file, output_format, args)
            else:
                cmd_analyze_audio_file(
                    audio_file, output_format, args)

        except Exception as e:
            output_format.error(e)
            has_error = True

    if has_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
