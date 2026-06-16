#!/usr/bin/env python3

import argparse
import sys

from typing import TextIO


def generate_beat_grid(
    output: TextIO,
    *,
    offset: float,
    bpm: float,
    duration: float,
    add_headings: bool = False,
):
    i = 0
    current = 0.0

    if add_headings:
        output.write("Time,Label\n")

    while True:
        current = offset + i * 60.0 / bpm
        if current >= duration + offset:
            break
        i += 1
        output.write(str(current) + "," + str(i) + "\n")


def main():
    has_error = False
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--offset",
        help="Start offset of the beat grid in seconds",
        type=float,
    )

    parser.add_argument(
        "-b", "--bpm",
        help="Generate a fixed-size beat grid with the given BPM (beats per minute).",
        type=float,
    )

    parser.add_argument(
        "-d", "--duration",
        help="Duration covered by the beat grid (in seconds).",
        type=float,
    )

    parser.add_argument(
        "--add-headings",
        dest="headings",
        help="Add column headings to the CSV file.",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "output_file",
        help="Path to output file",
        nargs='?'
    )

    args = parser.parse_args()

    if args.output_file:
        with open(args.output_file, "wt") as output_file:
            generate_beat_grid(
                output_file,
                offset=args.offset,
                bpm=args.bpm,
                duration=args.duration,
                add_headings=args.headings,
            )
    else:
        generate_beat_grid(
            sys.stdout,
            offset=args.offset,
            bpm=args.bpm,
            duration=args.duration,
            add_headings=args.headings,
        )

    if has_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
